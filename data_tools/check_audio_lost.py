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
dur_thre1 = 30
dur_thre2 = 0.5

# 将所有字典放入一个列表中
# datasets = NLP + HumanCentric + VideoQA + NaturalQA +VideoCap + OCRCap + NaturalCap
# datasets = NaturalCap + VideoCap + OCRCap + NaturalQA + VideoQA + HumanCentric + [TextSFT]
datasets = NaturalCap + VideoCap
datasets = OCRCap + NaturalQA
datasets = VideoQA + HumanCentric + [TextSFT]
datasets = [TextSFT]
# 初始化一个列表来存储丢失的文件名
lost_files = []
lock = threading.Lock()


def get_wav_duration(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    duration = waveform.size(1) / sample_rate
    return duration


def check_audio(audio_file_name, audio_directory):
    audio_file_path = os.path.join(audio_directory, "audio", audio_file_name)
    if not os.path.exists(audio_file_path):
        print(f"{audio_file_path} lost!!!!!!!!")
        return audio_file_name
    else:
        try:
            duration = get_wav_duration(audio_file_path)
            if duration > dur_thre1 or duration < dur_thre2:
                print(f"{audio_file_path} duration {duration}, too long!!!!!!!")
                return audio_file_name
        except Exception as e:
            print(f"{audio_file_path} is broken!!!!!!!!")
            return audio_file_name
    return None


# 遍历每个字典
for dataset in datasets:
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
            result = future.result()
            if result:
                with lock:
                    lost_files.append(result)

# 将丢失的文件名写入到lost_file_name.txt中
with open(output_file_path, "w", encoding="utf-8") as f:
    for file_name in lost_files:
        f.write(file_name + "\n")

print(f"检查完成，共有 {len(lost_files)} 个文件丢失或无法读取，结果已保存到 {output_file_path}")
