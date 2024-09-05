import onnxruntime
import torch
import numpy as np
import math
import sys
import os
import torchaudio
import pyaudio

import torchaudio.compliance.kaldi as k

class VADIterator:
    def __init__(self,
                 model,
                 threshold: float = 0.7,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 30
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):

        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = self.current_sample - self.speech_pad_samples - window_size_samples
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None

class WakeupAndVAD:
    def __init__(self, model_dir, keyword=None, cache_history=10, threshold=0.1):
        self.model_dir = model_dir
        self.chunk_size = 16
        self.chunk_overlap = 0
        self.feat_dim = 80
        self.frame_shift = 256
        self.CHUNK = self.frame_shift * self.chunk_size
        self.cache_history = cache_history

        self.keyword = keyword
        self.threshold = threshold

        self.is_wakeup = False
        self.in_dialog = False
        self.dialog_time = 0

        with torch.no_grad():

            self.load_vad()
            self.reset_dialog()
            self.history = torch.zeros(self.cache_history * 16000)
    
    def get_chunk_size(self):
        return self.CHUNK

    def load_model(self):
        self.sess_opt = onnxruntime.SessionOptions()
        self.sess_opt.intra_op_num_threads = 4
        self.sess_opt.inter_op_num_threads = 4

        sys.path.append(os.path.abspath(self.model_dir))

        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_shift , 1])

    def load_cmvn(self):
        cmvn_info = torch.load(f"{self.model_dir}/cmvn.dict")
        means = cmvn_info['mean_stat']
        variance = cmvn_info['var_stat']
        count = cmvn_info['frame_num']
        for i in range(len(means)):
            means[i] /= count
            variance[i] = variance[i] / count - means[i] * means[i]
            if variance[i] < 1.0e-20:
                variance[i] = 1.0e-20
            variance[i] = 1.0 / math.sqrt(variance[i])
        self.cmvn = np.array([means, variance]).astype(np.float32)
    
    def load_vad(self):
        self.vad_model = torch.jit.load(f"{self.model_dir}/silero_vad.jit")
        self.vad_iterator = VADIterator(self.vad_model)

        self.vad_model_post = torch.jit.load(f"{self.model_dir}/silero_vad.jit")
        self.vad_iterator_post = VADIterator(self.vad_model_post, min_silence_duration_ms=50)
    
    def reset_dialog(self):
        self.vad_iterator.reset_states()
        self.in_dialog = False
        self.dialog_time = 0
        self.dialog_part = torch.zeros([0,])
    
    def post_process_history(self, history):
        self.vad_iterator_post.reset_states()
        self.time_stamps = []
        for i in range(0, len(history) // 1024 * 1024, 1024):
            speech_dict = self.vad_iterator_post(history[i: i+ 1024], return_seconds=True)
            if speech_dict is not None and 'start' in speech_dict:
                self.time_stamps.append(speech_dict['start'])
        if self.cache_history - self.time_stamps[-1] < 1.5:
            history = history[:int(self.time_stamps[-1] * 16000)]
        return history

    def predict(self,
                audio: torch.Tensor):
        with torch.no_grad():
            audio = audio.clone().detach()
            speech_dict = self.vad_iterator(audio.reshape(-1), return_seconds=True)
            # print(speech_dict)
            if self.in_dialog:
                self.dialog_part = torch.cat([self.dialog_part, audio.reshape(-1)])
            if speech_dict is not None:
                if 'start' in speech_dict:
                    self.in_dialog = True
                    self.dialog_part = torch.cat([self.last_audio.reshape(-1), audio.reshape(-1)])
                    return speech_dict
                if self.in_dialog and 'end' in speech_dict:
                    output = {"cache_dialog": self.dialog_part.clone()}
                    self.reset_dialog()
                    self.is_wakeup = False
                    return output
            self.last_audio = audio.clone()
        return None
