
'''
import requests

def get_result_prompt(query):
    url = f"http://localhost:5001/api/search?query={query}&knowledgeLabel=1&reset=true"
    resp = requests.get(url)
    return resp.json().get("result_prompt")

# 示例
print(get_result_prompt("2022年春节档首日票房冠军"))


import json

import json

file_path = 'data/zh_refine.json'

# 逐行读取，每行一个 JSON 对象
with open(file_path, 'r', encoding='utf-8') as f:
    data_list = [json.loads(line) for line in f if line.strip()]

def create_text_content(data_list, use_positive=True, use_negative=True, positive_ratio=1.0, negative_ratio=1.0):
    lines = []
    for entry in data_list:
        lines.append(f'id: {entry["id"]}')
        if use_positive and "positive" in entry and entry["positive"]:
            num_pos = int(len(entry["positive"]) * positive_ratio)
            pos_samples = entry["positive"][:num_pos] if positive_ratio < 1.0 else entry["positive"]
            if pos_samples:
                lines.append('positive:')
                for p in pos_samples:
                    lines.append(p)
        if use_negative and "negative" in entry and entry["negative"]:
            num_neg = int(len(entry["negative"]) * negative_ratio)
            neg_samples = entry["negative"][:num_neg] if negative_ratio < 1.0 else entry["negative"]
            if neg_samples:
                lines.append('negative:')
                for n in neg_samples:
                    lines.append(n)
        lines.append('')  # 空行分隔
    return '\n'.join(lines)

# 1. 只保留正向数据
with open('positive_only.txt', 'w', encoding='utf-8') as f:
    f.write(create_text_content(data_list, use_positive=True, use_negative=False, positive_ratio=1.0))

# 2. 40%正向+60%负向混合
with open('mixed_40_60.txt', 'w', encoding='utf-8') as f:
    f.write(create_text_content(data_list, use_positive=True, use_negative=True, positive_ratio=0.4, negative_ratio=0.6))

# 3. 只保留负向数据
with open('negative_only.txt', 'w', encoding='utf-8') as f:
    f.write(create_text_content(data_list, use_positive=False, use_negative=True, negative_ratio=1.0))

print('All files generated successfully!')


# 假设你的txt文件名为 "example.txt"
filename = "mixed_40_60.txt"

with open(filename, 'r', encoding='utf-8') as file:
    content = file.read()
    word_count = len(content)

print(f"文件 '{filename}' 中的总字数为：{word_count}")
def count_positive_negative(filename):
    positive_count = 0
    negative_count = 0
    current_section = None

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('positive:'):
                current_section = 'positive'
                # 统计本行positive:后面的内容
                positive_count += len(stripped[len('positive:'):].strip())
            elif stripped.startswith('negative:'):
                current_section = 'negative'
                # 统计本行negative:后面的内容
                negative_count += len(stripped[len('negative:'):].strip())
            elif stripped.startswith('id:'):
                current_section = None
            else:
                if current_section == 'positive':
                    positive_count += len(stripped)
                elif current_section == 'negative':
                    negative_count += len(stripped)

    return positive_count, negative_count

if __name__ == "__main__":
    filename = "mixed_40_60.txt"
    pos_count, neg_count = count_positive_negative(filename)
    print(f"positive部分总字数：{pos_count}")
    print(f"negative部分总字数：{neg_count}")
'''

import os
import json

def merge_all_docs(instances, output_path):
    merged_content = []
    positive_chars = 0
    negative_chars = 0
    for instance in instances:
        # 合并所有positive
        if 'positive' in instance:
            pos = instance['positive']
            if isinstance(pos[0], list):
                for group in pos:
                    for doc in group:
                        merged_content.append(doc)
                        positive_chars += len(doc)
            else:
                for doc in pos:
                    merged_content.append(doc)
                    positive_chars += len(doc)
        # 合并所有negative
        if 'negative' in instance:
            for doc in instance['negative']:
                merged_content.append(doc)
                negative_chars += len(doc)
    total_chars = positive_chars + negative_chars
    positive_ratio = (positive_chars / total_chars * 100) if total_chars > 0 else 0
    negative_ratio = (negative_chars / total_chars * 100) if total_chars > 0 else 0

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(merged_content))
    print(f'合并完成 总字符数: {total_chars}')
    print(f'正样本比例: {positive_ratio:.2f}% ({positive_chars}字)')
    print(f'负样本比例: {negative_ratio:.2f}% ({negative_chars}字)')
    return output_path

if __name__ == "__main__":
    # 修改为你的数据集路径
    #dataset_path = 'data/zh_int.json'  # 或 'data/zh_int.json'
    dataset_path = 'data/zh_refine.json'
    output_path = 'merged_all_docs.txt'

    instances = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            instances.append(json.loads(line))

    merge_all_docs(instances, output_path)
