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
datasets = [Webvid]

# 遍历每个字典
for dataset in datasets:
    dur_list = []
    keys = list(dataset.keys())
    input_file_name = dataset["chat_path"]

    # 读取JSON文件
    with open(input_file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"check {input_file_name}")
    # 遍历每条数据
    for item in tqdm(data):
        # 是否有set_id
        assert "set" in item, f"{input_file_name} do not have set_id: {item} !!!!!!!!!!"

        # item是否为空
        assert len(item) > 0, f"{input_file_name} have null item!!!!!!!!!!"

        # 是否有键的值为空
        for key in item.keys():
            if type(item[key]) is not int and key != "id":
                assert (
                    len(item[key]) > 0
                ), f"{input_file_name}, item {item} have null key!!!!!!!!!!{key}"

        # item['conversations']是否有空
        for conv in item["conversations"]:
            text = conv["value"]
            if len(text) == 0:
                print(f"{input_file_name}, item {item} has null speaking!!!")

        # image/video路径数量、set_id数量、place_holder数量是否一致
        count_image_ph = 0
        count_video_ph = 0
        count_audio_ph = 0
        count_image_path = 0
        count_video_path = 0
        count_audio_path = 0

        text_all = ""
        for conv in item["conversations"]:
            text = conv["value"]
            text_all += text
            count_image_ph = text_all.count("<image>")
            count_video_ph = text_all.count("<video>")
            count_audio_ph = text_all.count("<audio>")

        if "image" in item:
            image_path = item["image"]
            assert isinstance(image_path[0], str)
            if type(image_path) is not list:
                assert isinstance(image_path, str)
                image_path = [image_path]
            count_image_path = len(image_path)

        if "video" in item:
            video_path = item["video"]
            assert isinstance(video_path[0], str)
            if type(video_path) is not list:
                assert isinstance(video_path, str)
                video_path = [video_path]
            count_video_path = len(video_path)

        if "audio" in item:
            audio_path = item["audio"]
            assert isinstance(audio_path[0], str)
            if type(audio_path) is not list:
                assert isinstance(audio_path, str)
                audio_path = [audio_path]
            count_audio_path = len(audio_path)

        # assert count_image_path == count_image_ph, f"{input_file_name}, item {item} image place holder number NOT equal image file number"
        # assert count_video_path == count_video_ph, f"{input_file_name}, item {item} video place holder number NOT equal video file number"
        # assert count_audio_path == count_audio_ph, f"{input_file_name}, item {item} audio place holder number NOT equal audio file number"

        if count_image_path != count_image_ph:
            print(
                f"{input_file_name}, item {item} image place holder number NOT equal image file number"
            )
        if count_video_path != count_video_ph:
            print(
                f"{input_file_name}, item {item} video place holder number NOT equal video file number"
            )
        if count_audio_path != count_audio_ph:
            print(
                f"{input_file_name}, item {item} audio place holder number NOT equal audio file number"
            )

        set_id = item["set"]
        if type(set_id) is not list:
            set_id = [set_id]
        if "image" in item or "video" in item:
            if set_id[0] != "sqa":
                assert (
                    len(set_id) == count_image_path + count_video_path
                ), f"{input_file_name}, item {item} set_id numer Not correct"
