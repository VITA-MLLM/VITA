import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

import torchaudio
from vita.config import *
from vita.config import AudioFolder, FolderDict
from vita.config.dataset_config import *

# 定义文件路径
output_file_path = "lost_file_name.txt"

# 将所有字典放入一个列表中
# datasets = NLP+HumanCentric+VideoQA+NaturalQA
datasets = VideoCap + OCRCap + NaturalCap

# 初始化一个列表来存储丢失的文件名
lock = threading.Lock()


def get_wav_duration(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


def check_audio(audio_file_name, audio_directory):
    audio_file_path = os.path.join(audio_directory, "audio", audio_file_name)
    duration = get_wav_duration(audio_file_path)
    if duration > 200:
        print(audio_file_path, duration)
    return duration


# 遍历每个字典
for dataset in datasets:
    dur_list = []
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]
    print(json_file_path)
    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历每条数据，使用tqdm显示进度条
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for item in data:
            audio_files = item.get("audio")
            audio_directory = AudioFolder
            # 如果 audio_files 是字符串，将其转换为列表
            if isinstance(audio_files, str):
                audio_files = [audio_files]

            # 如果 audio_files 是列表，处理每个文件
            if isinstance(audio_files, list):
                for audio_file_name in audio_files:
                    futures.append(executor.submit(check_audio, audio_file_name, audio_directory))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="file"
        ):
            duration = future.result()
            dur_list.append(duration)

    # 初始化区间计数字典
    distribution = {
        "0-1": 0,
        "1-5": 0,
        "5-10": 0,
        "10-15": 0,
        "15-20": 0,
        "20-25": 0,
        "25-30": 0,
        "30-60": 0,
        "60-200": 0,
        ">200": 0,
    }

    # 统计每个区间的计数
    for length in dur_list:
        if length <= 1:
            distribution["0-1"] += 1
        elif length <= 5:
            distribution["1-5"] += 1
        elif length <= 10:
            distribution["5-10"] += 1
        elif length <= 15:
            distribution["10-15"] += 1
        elif length <= 20:
            distribution["15-20"] += 1
        elif length <= 25:
            distribution["20-25"] += 1
        elif length <= 30:
            distribution["25-30"] += 1
        elif length <= 60:
            distribution["30-60"] += 1
        elif length <= 200:
            distribution["60-200"] += 1
        else:
            distribution[">200"] += 1

    # 打印分布结果
    print(f"duration distribution of {json_file_path}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")
