# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
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

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

import torchaudio
import torchaudio.compliance.kaldi as kaldi

from .adapter import CNNAdapter, CNNSubsampling, LinearAdapter
from .cmvn import GlobalCMVN, load_cmvn
from .module.encoder.encoder import whaleEncoder


class audioEncoderProcessor:
    def __init__(
        self,
        dataset_conf: dict = None,
    ):
        self.dataset_conf = dataset_conf

    def process(self, wav_path):
        try:
            waveform, sample_rate = torchaudio.load(wav_path)
        except Exception as e:
            print(f"cannot open {wav_path}!!!!!!!!!!!!!!!!")
        if sample_rate != self.dataset_conf["resample_conf"]["resample_rate"]:
            #            sample_rate = self.dataset_conf['resample_conf']['resample_rate']
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.dataset_conf["resample_conf"]["resample_rate"]
            )(waveform)

        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=self.dataset_conf["fbank_conf"]["num_mel_bins"],
            frame_length=self.dataset_conf["fbank_conf"]["frame_length"],
            frame_shift=self.dataset_conf["fbank_conf"]["frame_shift"],
            dither=self.dataset_conf["fbank_conf"]["dither"],
            energy_floor=0.0,
            sample_frequency=sample_rate,
        )
        attn_mask = torch.ones(mat.shape[0])
        attn_mask = attn_mask[2::2][2::2][0::2]

        return mat, attn_mask.shape[0]


class audioEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        llm_path: str,
        freeze_llm: bool = True,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 3,
        IGNORE_ID: int = -100,
        adpter_type: str = "cnn",
        add_audio_bos_eos: bool = False,
        task_num: int = 10,
        task_before_audio: bool = False,
        task_type: str = "prompt",
        freeze_encoder: bool = False,
        freeze_adpter: bool = False,
        activation_func: str = "relu",
        norm: str = "batch",
        chat_template=None,
    ):
        super().__init__()
        self.encoder = encoder

        self.enc_out_dim = enc_out_dim
        self.llm_embed_dim = llm_embed_dim
        self.IGNORE_ID = IGNORE_ID
        self.add_audio_bos_eos = add_audio_bos_eos
        self.task_before_audio = task_before_audio
        self.task_type = task_type
        self.freeze_encoder = freeze_encoder
        self.freeze_adpter = freeze_adpter

        if adpter_type == "cnn":
            self.adpter = CNNAdapter(enc_out_dim, llm_embed_dim, kernel_size)
        elif adpter_type == "linear":
            self.adpter = LinearAdapter(enc_out_dim, llm_embed_dim)
        elif adpter_type == "subsampling":
            self.adpter = CNNSubsampling(
                enc_out_dim, llm_embed_dim, kernel_size, activation_func, norm
            )

        if self.freeze_encoder:
            self.encoder.eval()
            for (name, param) in self.encoder.named_parameters():
                param.requires_grad = False
        if self.freeze_adpter:
            self.adpter.eval()
            for (name, param) in self.adpter.named_parameters():
                param.requires_grad = False

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:

        speech = speech.to(next(self.parameters()).dtype)

        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        inputs_embeds, encoder_mask = self.adpter(encoder_out, encoder_mask)  # B, T, D
        attention_mask = encoder_mask.squeeze(1)  # B, T
        assert inputs_embeds.size(1) == attention_mask.size(1)

        # audio bos/eos
        if self.add_audio_bos_eos:
            inputs_embeds, attention_mask, target = self._add_bos_eos(
                "audio", "/audio", inputs_embeds, attention_mask, target
            )

        outputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        return outputs

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None):
        B = len(inputs_embeds)
        bos_embed = self.task_embeddings(
            torch.full([B, 1], self.task_ids[bos]).to(inputs_embeds.device)
        )  # B, 1, D
        eos_embed = self.task_embeddings(
            torch.full([B, 1], self.task_ids[eos]).to(inputs_embeds.device)
        )  # B, 1, D
        bos_eos_target = torch.full([B, 2], self.IGNORE_ID).to(inputs_embeds.device)  # B, 2
        bos_eos_mask = torch.full([B, 1], True).to(inputs_embeds.device)  # B, 1

        inputs_embeds = torch.cat((bos_embed, inputs_embeds), 1)  # B, (1+T), D
        inputs_embeds = torch.cat((inputs_embeds, eos_embed), 1)  # B, (1+T+1), D
        attention_mask = torch.cat((bos_eos_mask, attention_mask), 1)  # B, (1+T)
        attention_mask = torch.cat((attention_mask, bos_eos_mask), 1)  # B, (1+T+1)
        if target is not None:
            target = torch.cat((target, bos_eos_target), 1)  # B, (T+2), D

        return inputs_embeds, attention_mask, target


def init_model(configs):
    if configs["cmvn_file"] is not None:
        mean, istd = load_cmvn(configs["cmvn_file"], configs["is_json_cmvn"])
        global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs["input_dim"]

    encoder = whaleEncoder(input_dim, global_cmvn=global_cmvn, **configs["encoder_conf"])
    model = audioEncoder(encoder=encoder, **configs["model_conf"])
    processor = audioEncoderProcessor(dataset_conf=configs["dataset_conf"])

    model.audio_processor = processor

    return model
