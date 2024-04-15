import json
import re
import os

def clean_description(description):
    """清理和转换描述字符串"""
    items = description.split('\n')
    cleaned_items = []
    for item in items:
        cleaned_item = re.sub(r'^\d+\.\s*', '', item)
        cleaned_item = cleaned_item[0].lower() + cleaned_item[1:]
        cleaned_item = re.sub(r'[,.]$', '', cleaned_item)
        cleaned_items.append(cleaned_item)
    return cleaned_items

def transform_data(file_path):
    """转换单个文件的数据"""
    with open(file_path, 'r') as infile:
        data = json.load(infile)
        
    transformed_data = {}
    for item in data:
        for compound, descriptions in item.items():
            transformed_data[compound] = clean_description(descriptions[0])
            
    # 覆盖原始文件
    with open(file_path, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

def transform_folder(folder_path):
    """转换指定文件夹下的所有 JSON 文件"""
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            print(f"Transforming {file_path}...")
            transform_data(file_path)
            print(f"Finished transforming {file_path}")

# 指定要转换的文件夹路径
folder_path = '/home/y/yangxi/proj/visualrep/code/word_soups/gpt_descriptors/fungi_descriptions'

# 调用函数开始转换过程
transform_folder(folder_path)