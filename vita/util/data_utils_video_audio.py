import copy
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from decord import VideoReader, cpu
from vita import conversation as conversation_lib
from vita.config import AudioFolder, DataConfig, FolderDict
from vita.constants import (
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    MAX_IMAGE_LENGTH,
    MIN_IMAGE_LENGTH,
)
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    dataset_use: str = field(default="temp")


def preprocess_multimodal(
    sources: Sequence[str], data_args: DataArguments, image_token_num=1, audio_lens: int = 0
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"] or DEFAULT_VIDEO_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"]
                    .replace(DEFAULT_IMAGE_TOKEN + "\n", DEFAULT_IMAGE_TOKEN)
                    .strip()
                )
                sentence["value"] = (
                    sentence["value"]
                    .replace("\n" + DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN)
                    .strip()
                )
                if sentence["value"].endswith(DEFAULT_IMAGE_TOKEN):
                    IMAGE_TOKEN_NUM = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, "").strip()
                    )
                    sentence["value"] = DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                if sentence["value"].endswith(DEFAULT_VIDEO_TOKEN):
                    VIDEO_TOKEN_NUM = sentence["value"].count(DEFAULT_VIDEO_TOKEN)
                    sentence["value"] = (
                        sentence["value"].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, "").strip()
                    )
                    sentence["value"] = DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM + sentence["value"]
                    sentence["value"] = sentence["value"].strip()

                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )

                IMAGE_TOKEN_NUM = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence["value"] = (
                        sentence["value"]
                        .replace(
                            DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                            DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH,
                        )
                        .strip()
                    )
            replace_token, vid_replace_token, audio_replace_token = (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_TOKEN * image_token_num,
                DEFAULT_AUDIO_TOKEN,
            )  # * audio_lens

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token + "\n")
            sentence["value"] = sentence["value"].replace(
                DEFAULT_VIDEO_TOKEN, vid_replace_token + "\n"
            )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_AUDIO_TOKEN + "\n", audio_replace_token
            )
            sentence["value"] = sentence["value"].replace("\n\n", "\n")

    return sources


def preprocess_mixtral_zh(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image and not has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    elif has_image and has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_audio_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    elif not has_image and has_audio:
        input_ids = torch.stack(
            [
                tokenizer_image_audio_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.MixtralZh

    # Mask targets
    sep = conv.sep + "\n" + conv.roles[1] + ":"
    sep2_2 = "\n" + conv.roles[0] + ":"
    sep2 = conv.sep2 + sep2_2
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(sep2)
        rounds = [rounds[0] + sep2 + rounds[1]] + rounds[2:]
        cur_len = 1
        end_token_cnt = 0
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break
            if i > 0:
                rou = sep2_2 + rou

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not has_audio:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            elif has_image and has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer)) - 1
            elif not has_image and has_audio:
                round_len = len(tokenizer_image_audio_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_audio_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            end_token_cnt += 1
            cur_len += round_len
        cur_len = cur_len - 1
        target[cur_len:] = IGNORE_INDEX

        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            cur_len -= end_token_cnt
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")
                # print(f"YOU NEED GO TO DEBUG THIS DATA ITEM: {conversations}")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    has_audio: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)

    if conversation_lib.default_conversation.version == "mixtral_zh":
        return preprocess_mixtral_zh(sources, tokenizer, has_image=has_image, has_audio=has_audio)


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=32,
    min_frames=4,
    image_resolution=384,
    video_framerate=1,
    s=None,
    e=None,
    image_aspect_ratio="pad",
):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    # T x 3 x H x W
    video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        elif len(all_pos) < min_frames:
            sample_pos = [
                all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

        if image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            patch_images = [
                expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean))
                for i in patch_images
            ]
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]
        else:
            patch_images = [
                image_processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                for i in patch_images
            ]

        # patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
        slice_len = len(patch_images)
        return patch_images, slice_len
        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            while len(patch_images) < max_frames:
                patch_images.append(torch.zeros((3, image_resolution, image_resolution)))
            # video[:slice_len, ...] = patch_images
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return patch_images, video_mask


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()

        dataset_list = DataConfig[str(data_args.dataset_use)]
        print(dataset_list)

        self.max_length = MAX_IMAGE_LENGTH
        list_data_dict = []
        self.folder_dict = {}
        for i in dataset_list:
            list_data_dict += json.load(open(i["chat_path"], "r"))

            image_folder = [folder for folder in i if folder is not "chat_path"]

            for folder in image_folder:
                if folder not in self.folder_dict:
                    self.folder_dict[folder] = i[folder]
        for key in FolderDict.keys():
            if key not in self.folder_dict:
                self.folder_dict[key] = FolderDict[key]

        random.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample or "video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0] and "audio" not in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            set_id = self.list_data_dict[i].get("set", None)
            file = image_file[0] if type(image_file) is list else image_file
            processor = self.data_args.image_processor

            if type(image_file) is list:
                assert type(set_id) is list
                if len(image_file) != len(set_id):
                    assert len(set(set_id)) == 1
                image = [
                    Image.open(
                        os.path.join(self.folder_dict[set_id[k]], file.replace("\\", "/"))
                    ).convert("RGB")
                    for k, file in enumerate(image_file)
                ]
                if self.data_args.image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [
                        expand2square(i, tuple(int(x * 255) for x in processor.image_mean))
                        for i in image
                    ]
                    image = [
                        processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in image
                    ]
                else:
                    image = [
                        processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in image
                    ]
            else:
                image_folder = self.folder_dict[set_id]
                image = Image.open(
                    os.path.join(image_folder, image_file.replace("\\", "/"))
                ).convert("RGB")
                if self.data_args.image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                else:
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )

            data_dict = preprocess(sources, self.tokenizer, has_image=True)

        elif "image" in sources[0] and "audio" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            set_id = self.list_data_dict[i].get("set", None)
            file = image_file[0] if type(image_file) is list else image_file
            audio_file = self.list_data_dict[i]["audio"]
            processor = self.data_args.image_processor

            if type(image_file) is list:
                assert type(set_id) is list
                if len(image_file) != len(set_id):  # 多图数据
                    assert len(set(set_id)) == 1
                image = [
                    Image.open(
                        os.path.join(self.folder_dict[set_id[k]], file.replace("\\", "/"))
                    ).convert("RGB")
                    for k, file in enumerate(image_file)
                ]
                if self.data_args.image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [
                        expand2square(i, tuple(int(x * 255) for x in processor.image_mean))
                        for i in image
                    ]
                    image = [
                        processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in image
                    ]
                else:
                    image = [
                        processor.preprocess(i, return_tensors="pt")["pixel_values"][0]
                        for i in image
                    ]
            else:
                image_folder = self.folder_dict[set_id]
                image = Image.open(
                    os.path.join(image_folder, image_file.replace("\\", "/"))
                ).convert("RGB")
                if self.data_args.image_aspect_ratio == "pad":

                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                else:
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

            if type(audio_file) is list:
                # if type(set_id) is list:
                #    audio_folder = self.folder_dict[set_id[0]+'_audio']
                # else:
                #    audio_folder = self.folder_dict[set_id+'_audio']
                audio_folder = AudioFolder
                assert len(audio_file) > 0, "audio_file为列表时不能为空"
                audio = []
                audio_for_llm_lens = []
                audio_length = []
                for file in audio_file:
                    try:
                        a, a_llm = self.data_args.audio_processor.process(
                            os.path.join(audio_folder, "audio", file)
                        )
                    except:
                        print(f"File {os.path.join(audio_folder, 'audio', file)} not OK!!!!!")
                    audio.append(a)
                    audio_for_llm_lens.append(a_llm)
                    audio_length.append(a.shape[0])
            else:
                # audio_folder = self.folder_dict[set_id+'_audio']
                audio_folder = AudioFolder
                assert audio_file, "audio_file不能为空"
                audio, audio_for_llm_lens = self.data_args.audio_processor.process(
                    os.path.join(audio_folder, "audio", audio_file)
                )
                audio_length = audio.shape[0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                audio_lens=audio_for_llm_lens,
            )

            data_dict = preprocess(sources, self.tokenizer, has_image=True, has_audio=True)
            data_dict["audio_lengths"] = audio_length
            data_dict["audio_lengths_for_llm"] = audio_for_llm_lens

        elif "video" in sources[0] and "audio" not in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_id = self.list_data_dict[i]["id"]
            set_id = self.list_data_dict[i].get("set", None)
            processor = self.data_args.image_processor
            if "height" in processor.size.keys():
                image_size = processor.size["height"]
            elif "shortest_edge" in processor.size.keys():
                image_size = processor.size["shortest_edge"]
            else:
                raise NotImplementedError(f"Please use correct key to use processor size!")
            video_folder = self.folder_dict[set_id]
            image, image_token_num = _get_rawvideo_dec(
                os.path.join(video_folder, video_file),
                processor,
                max_frames=MAX_IMAGE_LENGTH,
                min_frames=MIN_IMAGE_LENGTH,
                image_resolution=image_size,
            )

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                image_token_num=image_token_num,
            )

            data_dict = preprocess(sources, self.tokenizer, has_image=True, has_audio=False)

        elif "video" in sources[0] and "audio" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_id = self.list_data_dict[i]["id"]
            set_id = self.list_data_dict[i].get("set", None)
            audio_file = self.list_data_dict[i]["audio"]
            processor = self.data_args.image_processor
            if "height" in processor.size.keys():
                image_size = processor.size["height"]
            elif "shortest_edge" in processor.size.keys():
                image_size = processor.size["shortest_edge"]
            else:
                raise NotImplementedError(f"Please use correct key to use processor size!")
            video_folder = self.folder_dict[set_id]
            # audio_folder = self.folder_dict[set_id+'_audio']
            audio_folder = AudioFolder
            image, image_token_num = _get_rawvideo_dec(
                os.path.join(video_folder, video_file),
                processor,
                max_frames=MAX_IMAGE_LENGTH,
                min_frames=MIN_IMAGE_LENGTH,
                image_resolution=image_size,
            )
            if type(audio_file) is list:
                assert len(audio_file) > 0, "audio_file为列表时不能为空"
                audio = []
                audio_for_llm_lens = []
                audio_length = []
                for file in audio_file:
                    a, a_llm = self.data_args.audio_processor.process(
                        os.path.join(audio_folder, "audio", file)
                    )
                    audio.append(a)
                    audio_for_llm_lens.append(a_llm)
                    audio_length.append(a.shape[0])
            else:
                assert audio_file, "audio_file不能为空"
                audio, audio_for_llm_lens = self.data_args.audio_processor.process(
                    os.path.join(audio_folder, "audio", audio_file)
                )
                audio_length = audio.shape[0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                image_token_num=image_token_num,
                audio_lens=audio_for_llm_lens,
            )

            data_dict = preprocess(sources, self.tokenizer, has_image=True, has_audio=True)
            data_dict["audio_lengths"] = audio_length
            data_dict["audio_lengths_for_llm"] = audio_for_llm_lens
        elif "audio" in sources[0]:
            audio_file = self.list_data_dict[i]["audio"]
            audio_folder = AudioFolder
            if type(audio_file) is list:
                assert len(audio_file) > 0, "audio_file为列表时不能为空"
                audio = []
                audio_for_llm_lens = []
                audio_length = []
                for file in audio_file:
                    a, a_llm = self.data_args.audio_processor.process(
                        os.path.join(audio_folder, "audio", file)
                    )
                    audio.append(a)
                    audio_for_llm_lens.append(a_llm)
                    audio_length.append(a.shape[0])
            else:
                assert audio_file, "audio_file不能为空"
                audio, audio_for_llm_lens = self.data_args.audio_processor.process(
                    os.path.join(audio_folder, "audio", audio_file)
                )
                audio_length = audio.shape[0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                image_token_num=0,
                audio_lens=audio_for_llm_lens,
            )

            data_dict = preprocess(sources, self.tokenizer, has_image=False, has_audio=True)
            data_dict["audio_lengths"] = audio_length
            data_dict["audio_lengths_for_llm"] = audio_for_llm_lens
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

            data_dict = preprocess(sources, self.tokenizer, has_image=False)

        if isinstance(i, int):
            if "audio" in self.list_data_dict[i]:
                data_dict = dict(
                    input_ids=data_dict["input_ids"][0],
                    labels=data_dict["labels"][0],
                    audio_lengths=data_dict["audio_lengths"],
                    audio_lengths_for_llm=data_dict["audio_lengths_for_llm"],
                )
            else:
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i] or "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
        if "audio" in self.list_data_dict[i]:
            data_dict["audio"] = audio
        elif self.data_args.is_multimodal:
            data_dict["audio"] = torch.zeros(400, 80)
            data_dict["audio_lengths"] = 400
            data_dict["audio_lengths_for_llm"] = 60
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, : self.tokenizer.model_max_length]

        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        batch["audios"] = {}
        if "audio" in instances[0]:
            audios = [instance["audio"] for instance in instances]
            audio_lengths = [instance["audio_lengths"] for instance in instances]
            audio_lengths_for_llm = [instance["audio_lengths_for_llm"] for instance in instances]

            new_audios = []
            new_audio_lengths = []
            new_audio_lengths_for_llm = []
            for i, audio in enumerate(audios):
                length = audio_lengths[i]
                length_for_llm = audio_lengths_for_llm[i]
                if type(audio) is list:
                    for j, a in enumerate(audio):
                        new_audios.append(a)
                        new_audio_lengths.append(length[j])
                        new_audio_lengths_for_llm.append(length_for_llm[j])
                else:
                    new_audios.append(audio)
                    new_audio_lengths.append(length)
                    new_audio_lengths_for_llm.append(length_for_llm)
            audios = new_audios
            audios = pad_sequence(audios, batch_first=True, padding_value=0)

            batch["audios"]["audios"] = audios
            batch["audios"]["lengths"] = torch.tensor(new_audio_lengths)
            batch["audios"]["lengths_for_llm"] = torch.tensor(new_audio_lengths_for_llm)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
