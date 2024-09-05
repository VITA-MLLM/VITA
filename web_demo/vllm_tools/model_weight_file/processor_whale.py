# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for Speech2Text
"""

from typing import List, Optional, Union

import numpy as np
import os
import json

from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType, is_speech_available, logging


if is_speech_available():
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as ta_kaldi

logger = logging.get_logger(__name__)


class WhaleFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a WhaleFeatureExtractor for extracting features from raw speech.

    This feature extractor inherits from [`SequenceFeatureExtractor`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    This class extracts mel-filter bank features from raw speech using TorchAudio if installed or using numpy
    otherwise, and applies utterance-level cepstral mean and variance normalization (CMVN) to the extracted features.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized, expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the padding vectors.
        frame_length (`int`, *optional*, defaults to 25):
            The length of each frame in milliseconds.
        frame_shift (`int`, *optional*, defaults to 10):
            The shift between consecutive frames in milliseconds.
        dither (`float`, *optional*, defaults to 1.0):
            The amount of dithering (random noise) to apply to the signal.
        do_ceptral_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
        normalize_means (`bool`, *optional*, defaults to `True`):
            Whether or not to zero-mean normalize the extracted features.
        normalize_vars (`bool`, *optional*, defaults to `True`):
            Whether or not to unit-variance normalize the extracted features.
        cmvn_preload (`bool`, *optional*, defaults to `True`):
            Whether or not to preload CMVN statistics from a file.
        cmvn_file (`str`, *optional*, defaults to ""):
            Path to the file containing precomputed CMVN statistics.
        cmvn_means (`list` of `float`, *optional*, defaults to `None`):
            Precomputed means for CMVN.
        cmvn_istds (`list` of `float`, *optional*, defaults to `None`):
            Precomputed inverse standard deviations for CMVN.
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        frame_length=25,
        frame_shift=10,
        dither=1.0,
        do_ceptral_normalize=True,
        normalize_means=True,
        normalize_vars=True,
        cmvn_preload=True,
        cmvn_file="",
        cmvn_means=None,
        cmvn_istds=None,
        **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.num_mel_bins = num_mel_bins
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.dither = dither
        self.do_ceptral_normalize = do_ceptral_normalize
        self.normalize_means = normalize_means
        self.normalize_vars = normalize_vars
        self.return_attention_mask = True
        self.cmvn_preload = cmvn_preload
        self.cmvn_file = cmvn_file
        self.cmvn_means = cmvn_means
        self.cmvn_istds = cmvn_istds

        if self.cmvn_preload:
            if self.cmvn_means is not None and self.cmvn_istds is not None:
                self.cmvn_means = np.array(self.cmvn_means, dtype=np.float32)
                self.cmvn_istds = np.array(self.cmvn_istds, dtype=np.float32)
            else:
                if self.cmvn_file is None or self.cmvn_file == "":
                    raise ValueError(f"cmvn_file should be a valid file if cmvn_preload is set True, but we get {self.cmvn_file}.")
                if not os.path.join(self.cmvn_file):
                    raise ValueError(f"file {self.cmvn_file} is not found.")
                self.cmvn_means, self.cmvn_istds = self._load_json_cmvn(self.cmvn_file)

        if not is_speech_available():
            mel_filters = mel_filter_bank(
                num_frequency_bins=256,
                num_mel_filters=self.num_mel_bins,
                min_frequency=20,
                max_frequency=sampling_rate // 2,
                sampling_rate=sampling_rate,
                norm=None,
                mel_scale="kaldi",
                triangularize_in_mel_space=True,
            )

            self.mel_filters = np.pad(mel_filters, ((0, 1), (0, 0)))
            self.window = window_function(400, "povey", periodic=False)

    def _load_json_cmvn(self, json_cmvn_file):
        """ Load the json format cmvn stats file and calculate cmvn

        Args:
            json_cmvn_file: cmvn stats file in json format

        Returns:
            a numpy array of [means, vars]
        """
        with open(json_cmvn_file) as f:
            cmvn_stats = json.load(f)

        means = np.array(cmvn_stats['mean_stat'])
        variances = np.array(cmvn_stats['var_stat'])
        count = cmvn_stats['frame_num']

        epsilon = 1.0e-6

        means = means / count
        variances = variances / count - means ** 2
        variances[variances < epsilon] = epsilon
        istds = 1.0 / np.sqrt(variances)

        return means, istds


    def _extract_fbank_features(
        self,
        waveform: np.ndarray,
    ) -> np.ndarray:
        """
        Get mel-filter bank features using TorchAudio. Note that TorchAudio requires 16-bit signed integers as inputs
        and hence the waveform should not be normalized before feature extraction.
        """
        waveform = waveform * (2**15)  # Kaldi compliance: 16-bit signed integers
        if is_speech_available():
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform)
            
            features = ta_kaldi.fbank(
                waveform,
                num_mel_bins=self.num_mel_bins,
                sample_frequency=self.sampling_rate,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
                dither=self.dither,
                energy_floor=0.0,
            )
            features = features.numpy()
        else:
            waveform = np.squeeze(waveform)
            features = spectrogram(
                waveform,
                self.window,
                frame_length=400,
                hop_length=160,
                fft_length=512,
                power=2.0,
                center=False,
                preemphasis=0.97,
                mel_filters=self.mel_filters,
                log_mel="log",
                mel_floor=1.192092955078125e-07,
                remove_dc_offset=True,
            ).T
        return features

    @staticmethod
    def utterance_cmvn(
        x: np.ndarray,
        input_length: int,
        normalize_means: Optional[bool] = True,
        normalize_vars: Optional[bool] = True,
        padding_value: float = 0.0,
        cmvn_means: Optional[np.ndarray] = None,
        cmvn_istds: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # make sure we normalize float32 arrays
        if normalize_means:
            mean = cmvn_means if cmvn_means is not None else x[:input_length].mean(axis=0)
            x = np.subtract(x, mean)
        if normalize_vars:
            istd = cmvn_istds if cmvn_istds is not None else 1 / x[:input_length].std(axis=0)
            x = np.multiply(x, istd)

        if input_length < x.shape[0]:
            x[input_length:] = padding_value

        # make sure array is in float32
        x = x.astype(np.float32)

        return x

    def normalize(
        self, input_features: List[np.ndarray], attention_mask: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        lengths = attention_mask.sum(-1) if attention_mask is not None else [x.shape[0] for x in input_features]
        return [
            self.utterance_cmvn(
                x, 
                n, 
                self.normalize_means, 
                self.normalize_vars, 
                self.padding_value,
                self.cmvn_means if self.cmvn_preload else None,
                self.cmvn_istds if self.cmvn_preload else None,
            )
            for x, n in zip(input_features, lengths)
        ]

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
                stereo, i.e. single float per timestep.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`):
                Activates truncation to cut input sequences longer than *max_length* to *max_length*.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
            return_attention_mask (`bool`, *optional*):
                Whether to return the attention mask. If left to the default, will return the attention mask according
                to the specific feature_extractor's default.

                [What are attention masks?](../glossary#attention-mask)

                <Tip>

                For Speech2TextTransformer models, `attention_mask` should always be passed for batched inference, to
                avoid subtle bugs.

                </Tip>

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            padding_value (`float`, defaults to 0.0):
                The value that is used to fill the padding values / vectors.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                logger.warning(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
                if is_speech_available():
                    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.sampling_rate)
                    if isinstance(raw_speech, List):
                        raw_speech = [resampler(speech) for speech in raw_speech]
                    else:
                        raw_speech = resampler(raw_speech)

                logger.warning(
                    f"Resampling the input audio to match the model's sampling rate of {self.sampling_rate}."
                )

        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features
        features = [self._extract_fbank_features(waveform) for waveform in raw_speech]

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )

        # make sure list is in array format
        input_features = padded_inputs.get("input_features")
        if isinstance(input_features[0], list):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]

        attention_mask = padded_inputs.get("attention_mask")
        if attention_mask is not None:
            padded_inputs["attention_mask"] = [np.asarray(array, dtype=np.int32) for array in attention_mask]

        # Utterance-level cepstral mean and variance normalization
        if self.do_ceptral_normalize:
            attention_mask = (
                np.array(attention_mask, dtype=np.int32)
                if self._get_padding_strategies(padding, max_length=max_length) is not PaddingStrategy.DO_NOT_PAD
                else None
            )
            padded_inputs["input_features"] = self.normalize(
                padded_inputs["input_features"], attention_mask=attention_mask
            )

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

