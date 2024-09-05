import json

from vita.constants import GLOBAL_WEIGHTS_PATH

# 定义文件路径
lost_file_path = "lost_file_name.txt"
json_list = []

for json_file_path in json_list:
    output_json_file_path = json_file_path

    with open(lost_file_path, "r") as file:
        lost_files = set(file.read().splitlines())

    # Load the JSON data
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # 过滤数据，删除丢失文件对应的数据
    filtered_data = []
    for item in data:
        video_OK = True
        if "video" in item:
            video_filename = item["video"]
            if video_filename in lost_files:
                video_OK = False
        if video_OK:
            filtered_data.append(item)

    # Save the filtered data back to a new JSON file
    with open(output_json_file_path, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=2, ensure_ascii=False)

    print(
        f"The json data has been delete {len(data)-len(filtered_data)} and saved to {output_json_file_path}"
    )
