import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from decord import VideoReader, cpu
from vita.config import FolderDict
from vita.config.dataset_config import *

# 定义文件路径
output_file_path = "lost_file_name.txt"

# 将所有字典放入一个列表中
# datasets = [Webvid, K400]
# datasets = [VIDEOChatGPT, K700Split, VC2Internvid]
# datasets = [EgoGesture, Literature, CopyWrite, MovingFashion]
# datasets = [NoHarm]
datasets = [SGInternvid0]

# 初始化一个列表来存储丢失的文件名
lost_files = []

# 遍历每个字典
for dataset in datasets:
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def check_video_file(item):
        video_file_name = item.get("video")
        if video_file_name:
            video_directory = FolderDict[item["set"]]
            video_file_path = os.path.join(video_directory, video_file_name)
            if not os.path.exists(video_file_path):
                print(f"file lost: {video_file_path}")
                return video_file_name
            else:
                sample_pos = [0, 10]
                try:
                    vreader = VideoReader(video_file_path, ctx=cpu(0))
                    patch_images = [
                        Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()
                    ]
                except Exception as e:
                    print(f"file broken: {video_file_path}")
                    return video_file_name
        return None

    # 使用ThreadPoolExecutor进行多线程并行处理
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(check_video_file, item) for item in data]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="file"
        ):
            result = future.result()
            if result:
                lost_files.append(result)

# 将丢失的文件名写入到lost_file_name.txt中
with open(output_file_path, "w", encoding="utf-8") as f:
    for file_name in lost_files:
        f.write(file_name + "\n")

print(f"检查完成，共有 {len(lost_files)} 个文件丢失或无法读取，结果已保存到 {output_file_path}")
