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
from vita.constants import AUDIO_TOKEN_INDEX, GLOBAL_WEIGHTS_PATH, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vita.util.data_utils_video_audio import DataArguments, LazySupervisedDataset
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token

image_token_num = 256
token_thre = 4500
# datasets = NLP + HumanCentric + VideoQA + NaturalQA + VideoCap + OCRCap + NaturalCap
datasets = [DyChart_iresearch, RCTW2019QA, Lvis_cn_noDesc, VIDEOChatGPT]
datasets = [AnyWord_20to50]
out_file_name = "debug.json"

parser = transformers.HfArgumentParser((DataArguments))
tokenizer = transformers.AutoTokenizer.from_pretrained(
    f"{GLOBAL_WEIGHTS_PATH}/Mixtral-8x7B_New/mg2hg",
    cache_dir=None,
    model_max_length=8192,
    padding_side="right",
    use_fast=True,
)

long_json = []


def get_wav_duration(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


def process_item(item, conv, roles, tokenizer):
    source = item["conversations"]
    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{source}"
        conv.append_message(role, sentence["value"])
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
    item_token_num = input_ids.shape[0] + num_images * image_token_num

    if "audio" in item:
        audio_files = item["audio"]
        audio_directory = AudioFolder
        if isinstance(audio_files, str):
            audio_files = [audio_files]
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
    if item_token_num > token_thre:
        print(item_token_num)
        if len(item["image"]) >= 16:
            long_json.append(item)
            print(len(item["image"]))
    return item_token_num


for dataset in datasets:
    json_file_path = dataset["chat_path"]

    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    len_list = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_item, item, conv, roles, tokenizer) for item in data]
        for future in tqdm(as_completed(futures), total=len(futures)):
            len_list.append(future.result())

    assert len(len_list) == len(data)

    distribution = {
        "0-100": 0,
        "100-200": 0,
        "200-300": 0,
        "300-400": 0,
        "400-500": 0,
        "500-600": 0,
        "600-700": 0,
        "700-800": 0,
        "800-900": 0,
        "900-1000": 0,
        "1000-1100": 0,
        "1100-1200": 0,
        "1200-1300": 0,
        "1300-1400": 0,
        "1400-1500": 0,
        "1500-1600": 0,
        "1600-1700": 0,
        "1700-1800": 0,
        "1800-1900": 0,
        "1900-2000": 0,
        "2000-2500": 0,
        "2500-3000": 0,
        "3000-3500": 0,
        "3500-4000": 0,
        "4000-4500": 0,
        "4500-5000": 0,
        "5000-5500": 0,
        "5500-6000": 0,
        ">6000": 0,
    }

    for length in len_list:
        if length <= 100:
            distribution["0-100"] += 1
        elif length <= 200:
            distribution["100-200"] += 1
        elif length <= 300:
            distribution["200-300"] += 1
        elif length <= 400:
            distribution["300-400"] += 1
        elif length <= 500:
            distribution["400-500"] += 1
        elif length <= 600:
            distribution["500-600"] += 1
        elif length <= 700:
            distribution["600-700"] += 1
        elif length <= 800:
            distribution["700-800"] += 1
        elif length <= 900:
            distribution["800-900"] += 1
        elif length <= 1000:
            distribution["900-1000"] += 1
        elif length <= 1100:
            distribution["1000-1100"] += 1
        elif length <= 1200:
            distribution["1100-1200"] += 1
        elif length <= 1300:
            distribution["1200-1300"] += 1
        elif length <= 1400:
            distribution["1300-1400"] += 1
        elif length <= 1500:
            distribution["1400-1500"] += 1
        elif length <= 1600:
            distribution["1500-1600"] += 1
        elif length <= 1700:
            distribution["1600-1700"] += 1
        elif length <= 1800:
            distribution["1700-1800"] += 1
        elif length <= 1900:
            distribution["1800-1900"] += 1
        elif length <= 2000:
            distribution["1900-2000"] += 1
        elif length <= 2500:
            distribution["2000-2500"] += 1
        elif length <= 3000:
            distribution["2500-3000"] += 1
        elif length <= 3500:
            distribution["3000-3500"] += 1
        elif length <= 4000:
            distribution["3500-4000"] += 1
        elif length <= 4500:
            distribution["4000-4500"] += 1
        elif length <= 5000:
            distribution["4500-5000"] += 1
        elif length <= 5500:
            distribution["5000-5500"] += 1
        elif length <= 6000:
            distribution["5500-6000"] += 1
        else:
            distribution[">6000"] += 1

    print(f"Length distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")

# with open(out_file_name, 'w', encoding='utf-8') as file:
#    json.dump(long_json*10, file, ensure_ascii=False, indent=4)

# print(f"处理完成，大于{token_thre}的已保存到{out_file_name}")
