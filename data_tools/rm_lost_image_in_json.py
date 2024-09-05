import json

from vita.constants import GLOBAL_WEIGHTS_PATH

# 定义文件路径
# lost_file_path = 'lost_file_name.txt'
lost_file_path = "long_image_file_name.txt"
json_list = [""]

for json_file_path in json_list:
    output_json_file_path = json_file_path
    # 读取丢失的文件名
    with open(lost_file_path, "r", encoding="utf-8") as f:
        lost_files = set(line.strip() for line in f)

    # 读取JSON文件
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 过滤数据，删除丢失文件对应的数据
    filtered_data = []
    for item in data:
        image_OK = True
        if "image" in item:
            image_file = item["image"]
            if type(image_file) is str:
                image_file = [image_file]
            assert type(image_file) is list
            for image_filename in image_file:
                if image_filename in lost_files:
                    image_OK = False
                    break
        if image_OK:
            filtered_data.append(item)

    # 将更新后的数据写入新的JSON文件
    with open(output_json_file_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"更新完成，共删除了 {len(data) - len(filtered_data)} 条数据，结果已保存到 {output_json_file_path}")
