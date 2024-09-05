import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import transformers
from tqdm import tqdm

import torchaudio
from vita import conversation as conversation_lib
from vita.config import *
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *
from vita.constants import AUDIO_TOKEN_INDEX, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vita.util.data_utils_video_audio import DataArguments, LazySupervisedDataset
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token

image_token_num = 256
token_thre = 4500
# datasets = NaturalCap + VideoCap + OCRCap + NaturalQA + VideoQA + HumanCentric + NLP
datasets = (
    NaturalCap0
    + OCRCap0
    + VideoCap0
    + NaturalQA0
    + VideoQA0
    + [EgoGesture0, Literature0, CopyWrite0, MovingFashion0]
)

num_data_neg_audio = 0
for dataset in datasets:
    json_file_path = dataset["chat_path"]

    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    num_data_audio = 0
    num_data_conv = 0
    num_data_qs_qudio = 0
    num_data_qs_text = 0
    for item in data:
        conversations = item["conversations"]
        assert len(conversations) % 2 == 0
        num_conv = len(conversations) // 2
        num_data_conv += num_conv
        num_qs_audio = 0
        num_qs_text = 0
        for conv in conversations:
            if conv["from"] == "human":
                qs = conv["value"]
                if "<audio>" in qs:
                    num_qs_audio += 1
                else:
                    num_qs_text += 1
        num_data_qs_qudio += num_qs_audio
        num_data_qs_text += num_qs_text
        num_audio = 0
        audio_files = item.get("audio")
        audio_directory = AudioFolder
        # 如果 audio_files 是字符串，将其转换为列表
        if isinstance(audio_files, str):
            audio_files = [audio_files]

        # 如果 audio_files 是列表，处理每个文件
        if isinstance(audio_files, list):
            num_audio = len(audio_files)
            for audio in audio_files:
                if "new_value_dict_0725" in audio or "new_value_dict_0730" in audio:
                    num_data_neg_audio += 1
        num_data_audio += num_audio

    assert num_data_conv == num_data_qs_qudio + num_data_qs_text
    # print(f'{json_file_path} conversation number: {num_data_conv/1000}K')
    # print(f'{json_file_path} audio question number: {num_data_qs_qudio/1000}K')
    # print(f'{json_file_path} text question number: {num_data_qs_text/1000}K')
    # print(f'{json_file_path} audio number: {num_data_audio/1000}K')
    print(f"{json_file_path} data number: {len(data)/1000}K")

# print(f'{json_file_path} negtive audio question number: {num_data_neg_audio/1000}K')
