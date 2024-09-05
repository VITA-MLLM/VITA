import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import transformers
from PIL import Image
from tqdm import tqdm

import torchaudio
from vita import conversation as conversation_lib
from vita.config import *
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *
from vita.constants import AUDIO_TOKEN_INDEX, GLOBAL_WEIGHTS_PATH, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from vita.util.data_utils_video_audio import DataArguments, LazySupervisedDataset
from vita.util.data_utils_video_audio_neg_patch import find_closest_aspect_ratio
from vita.util.mm_utils import tokenizer_image_audio_token, tokenizer_image_token

image_token_num = 256
token_thre = 9500
# datasets = NLP + HumanCentric + VideoQA + NaturalQA + VideoCap + OCRCap + NaturalCap

datasets = NaturalCap + OCRCap + VideoCap + NaturalQA
# datasets = VideoQA + HumanCentric + NLP
# datasets = [SGInternvid0]
datasets = [TextSFT, TextSFT2_0]

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


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    if use_thumbnail and blocks != 1:
        blocks += 1
    return blocks


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
    item_token_num = input_ids.shape[0]
    if "image" in item:
        image_file = item["image"]
        if isinstance(image_file, str):
            image_file = [image_file]
        set_id = item["set"]
        if isinstance(set_id, str):
            set_id = [set_id]
        for k, img_file in enumerate(image_file):
            if set_id[k] not in NoPatchSets:
                image_directory = FolderDict[set_id[k]]
                image = Image.open(
                    os.path.join(image_directory, img_file.replace("\\", "/"))
                ).convert("RGB")
                num_patches = dynamic_preprocess(image)
            else:
                num_patches = 1
            item_token_num += num_patches * image_token_num

    total_duration = 0
    if "audio" in item:
        audio_files = item["audio"]
        audio_directory = AudioFolder
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        assert isinstance(audio_files, list)
        for audio_file_name in audio_files:
            audio_file_path = os.path.join(audio_directory, "audio", audio_file_name)
            duration = get_wav_duration(audio_file_path)
            duration = (
                math.ceil(duration) if math.ceil(duration) % 2 == 0 else math.ceil(duration) + 1
            )
            total_duration += duration
        item_token_num += math.ceil(total_duration * 12.5)
    if item_token_num > token_thre:
        print(f"item_token_num: {item_token_num}")
        if len(item["image"]) >= 16:
            print(f"num_patches: {num_patches}")
            print(f"total_duration: {total_duration}")
            long_json.append(item)
            print(item)
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
        "1000-1500": 0,
        "1500-2000": 0,
        "2000-2500": 0,
        "2500-3000": 0,
        "3000-3500": 0,
        "3500-4000": 0,
        "4000-4500": 0,
        "4500-5000": 0,
        "5000-5500": 0,
        "5500-6000": 0,
        "6000-6500": 0,
        "6500-7000": 0,
        "7000-7500": 0,
        "7500-8000": 0,
        "8000-8500": 0,
        "8500-9000": 0,
        "9000-9500": 0,
        "9500-10000": 0,
        ">10000": 0,
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
        elif length <= 1500:
            distribution["1000-1500"] += 1
        elif length <= 2000:
            distribution["1500-2000"] += 1
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
        elif length <= 6500:
            distribution["6000-6500"] += 1
        elif length <= 7000:
            distribution["6500-7000"] += 1
        elif length <= 7500:
            distribution["7000-7500"] += 1
        elif length <= 8000:
            distribution["7500-8000"] += 1
        elif length <= 8500:
            distribution["8000-8500"] += 1
        elif length <= 9000:
            distribution["8500-9000"] += 1
        elif length <= 9500:
            distribution["9000-9500"] += 1
        elif length <= 10000:
            distribution["9500-10000"] += 1
        else:
            distribution[">10000"] += 1

    print(f"Length distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")

# with open(out_file_name, 'w', encoding='utf-8') as file:
#    json.dump(long_json*10, file, ensure_ascii=False, indent=4)

# print(f"处理完成，大于{token_thre}的已保存到{out_file_name}")
