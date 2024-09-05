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

# 将所有字典放入一个列表中
datasets = NLP + HumanCentric + VideoQA + NaturalQA + VideoCap + OCRCap + NaturalCap

# 遍历每个字典
for dataset in datasets:
    dur_list = []
    keys = list(dataset.keys())
    input_file_name = dataset["chat_path"]

    # 读取JSON文件
    len_list = []
    with open(input_file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"check {input_file_name}")
    # 遍历每条数据
    for item in tqdm(data):
        if "image" in item:
            image_path = item["image"]
            assert isinstance(image_path[0], str)
            if type(image_path) is not list:
                assert isinstance(image_path, str)
                image_path = [image_path]
            count_image_path = len(image_path)
            if count_image_path > 40:
                print(count_image_path)
                print(item)
            len_list.append(count_image_path)

    distribution = {
        "0-5": 0,
        "5-10": 0,
        "10-16": 0,
        "16-20": 0,
        "20-25": 0,
        "25-30": 0,
        "30-35": 0,
        "35-40": 0,
        ">40": 0,
    }

    for length in len_list:
        if length <= 5:
            distribution["0-5"] += 1
        elif length <= 10:
            distribution["5-10"] += 1
        elif length <= 16:
            distribution["10-16"] += 1
        elif length <= 20:
            distribution["16-20"] += 1
        elif length <= 25:
            distribution["20-25"] += 1
        elif length <= 30:
            distribution["25-30"] += 1
        elif length <= 35:
            distribution["30-35"] += 1
        elif length <= 40:
            distribution["35-40"] += 1
        else:
            distribution[">40"] += 1

    print(f"Length distribution of {input_file_name}:")
    for key, value in distribution.items():
        print(f"{key}: {value}")
