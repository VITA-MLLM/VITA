import json
import math
import os
import random

import torch
import transformers
from tqdm import tqdm

import torchaudio
from vita import conversation as conversation_lib
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *
from vita.constants import AUDIO_TOKEN_INDEX, GLOBAL_WEIGHTS_PATH, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vita.util.data_utils_video_audio import DataArguments, LazySupervisedDataset
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token

image_token_num = 256
concat_size = 4500
datasets = [ShareGPT4V]

parser = transformers.HfArgumentParser((DataArguments))
tokenizer = transformers.AutoTokenizer.from_pretrained(
    f"{GLOBAL_WEIGHTS_PATH}/Mixtral-8x7B_New/mg2hg",
    cache_dir=None,
    model_max_length=8192,
    padding_side="right",
    use_fast=True,
)


def get_wav_duration(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


for dataset in datasets:
    input_file_name = dataset["chat_path"]
    base_name, ext = os.path.splitext(input_file_name)
    suffix = f"-concat{concat_size}"
    out_file_name = f"{base_name}{suffix}{ext}"

    with open(input_file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    random.shuffle(data)
    # data = data[:100]

    # 遍历每条数据
    len_list = []

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    len_list = []
    # Apply prompt templates
    for item in tqdm(data):
        source = item["conversations"]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{source}"
            conv.append_message(role, sentence["value"])
        prompt = conv.get_prompt()

        # import pdb; pdb.set_trace()
        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
        item_token_num = input_ids.shape[0] + num_images * image_token_num

        if "audio" in item:
            audio_files = item["audio"]
            audio_directory = AudioFolder
            # 如果 audio_files 是字符串，将其转换为列表
            if isinstance(audio_files, str):
                audio_files = [audio_files]
            # 如果 audio_files 是列表，处理每个文件
            assert isinstance(audio_files, list)
            total_duration = 0
            for audio_file_name in audio_files:
                audio_file_path = os.path.join(audio_directory, "audio", audio_file_name)
                duration = get_wav_duration(audio_file_path)
                duration = (
                    math.ceil(duration) if math.ceil(duration) % 2 == 0 else math.ceil(duration) + 1
                )
                total_duration += duration
            item_token_num += math.ceil(total_duration * 12.5)
        len_list.append(item_token_num)
    assert len(len_list) == len(data)

    def concat_item(items):
        temp_set_id = []
        temp_conversations = []
        temp_ids = []
        temp_images = []
        temp_audios = []

        for item in items:
            temp_set_id.append(item["set"])
            temp_conversations.extend(item["conversations"])
            if "id" in item:
                temp_ids.append(item["id"])
            if "image" in item:
                temp_images.append(item["image"])
            if "audio" in item:
                audio = item["audio"]
                if type(audio) is not list:
                    audio = [audio]
                temp_audios += audio
        if len(temp_images) > 0:
            merged_item = {
                "set": temp_set_id,
                "id": temp_ids,
                "image": temp_images,
                "conversations": temp_conversations,
            }
        else:
            merged_item = {
                "set": temp_set_id,
                "id": temp_ids,
                "conversations": temp_conversations,
            }
        if len(temp_audios) > 0:
            merged_item["audio"] = temp_audios
        return merged_item

    merged_data = []
    i = 0
    while i < len(data):
        len_token = len_list[i]
        k = 1
        while True:
            if sum(len_list[i : i + k]) > concat_size:
                if k > 1:
                    k -= 1
                break
            if i + k == len(data):
                break
            k += 1
        merged_item = concat_item(data[i : i + k])
        merged_data.append(merged_item)
        #    print(f"i: {i}, k: {k}; len of merged item: {sum(len_list[i:i+k])}")
        i = i + k

    with open(out_file_name, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"save {out_file_name}")
