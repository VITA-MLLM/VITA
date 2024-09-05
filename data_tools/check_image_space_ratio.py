import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

from vita.config import FolderDict
from vita.config.dataset_config import *

# 定义文件路径
output_file_path = "long_image_file_name.txt"
ratio_thre = 12

# 将所有字典放入一个列表中
# datasets = [AnyWord_20to50, RCTW2019, RCTW2019QA, RCTW2017, OpenChart, SCID, K12, TabRECSet, DigChat, iFlyTab]
datasets = [AnyWord_20to50, DyChart_iresearch]
datasets = [RCTW2019, RCTW2019QA, RCTW2017]
datasets = [OpenChart, SCID]
datasets = [K12]
# datasets = [TabRECSet, DigChat, iFlyTab]

# 初始化一个列表来存储丢失的文件名
lost_files = []
lock = threading.Lock()


def check_image(image_file_name, image_directory):
    image_file_path = os.path.join(image_directory, image_file_name)
    if not os.path.exists(image_file_path):
        print(f"{image_file_path} not exist!!!!!!!!!!")
        return image_file_name
    else:
        try:
            with Image.open(image_file_path) as img:
                img.convert("RGB")
                size_ratio = img.size[0] / img.size[1]
                if size_ratio < 1 / ratio_thre or size_ratio > ratio_thre:
                    print(f"{image_file_path} ratio is too big!!!!!!!!!!!!!!")
                    return image_file_name
        except Exception as e:
            print(f"{image_file_path} is broken!!!!!!!!!!!!")
            return image_file_name
    return None


# 遍历每个字典
for dataset in datasets:
    keys = list(dataset.keys())
    json_file_path = dataset["chat_path"]

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 遍历每条数据，使用tqdm显示进度条
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for item in data:
            image_files = item.get("image")
            set_id = item["set"]
            image_directory = FolderDict[set_id]

            # 如果 image_files 是字符串，将其转换为列表
            if isinstance(image_files, str):
                image_files = [image_files]

            # 如果 image_files 是列表，处理每个文件
            if isinstance(image_files, list):
                for image_file_name in image_files:
                    futures.append(executor.submit(check_image, image_file_name, image_directory))

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
