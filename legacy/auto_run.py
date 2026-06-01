#!/usr/bin/env python3
import subprocess

# 定义模型列表
models = [
    'deepseek-r1:1.5b',
    'deepseek-r1:7b',
    'deepseek-r1:8b',
    'qwen3:0.6b',
    'qwen3:1.7b',
    'qwen3:4b',
    'qwen3:8b'
]

# 定义数据集列表
#datasets = ['zh_refine']
datasets = ['zh_int']
# 遍历所有数据集和模型的组合
for dataset in datasets:
    for model in models:
        print(f"正在运行 - 数据集: {dataset}, 模型: {model}")
        # 构造运行命令
        command = ["python", "new_test1.py", "--modelname", model, "--dataset", dataset]
        # 执行命令
        subprocess.run(command)