
# deepseekMine检索测试与生成性能测试说明

本说明文档详细介绍了deepseekMine检索测试与生成性能测试的操作流程，涵盖逐步上传与一次性上传两种模式，并附带环境安装说明，便于快速上手。

## 环境准备

1. **安装依赖环境**
    - 请确保已安装 Python 3.10 及以上版本。
    - 使用如下命令安装依赖包：

```bash
pip install -r requirements.txt
```

## 检索测试操作说明

### 1. 逐步上传模式

- **测试脚本：** `rag_retrieval_eval.py`
- **参数说明：**
    - `--dataset`：数据集名称（`zh_int` 或 `zh_refine`）
    - `--noise_rate`：噪声比率（如 0.0、0.2、0.4、0.6、0.8）
    - `--num`：每组测试数量
- **示例代码：**

```python
import os

noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
for rate in noise_rates:
    cmd = f"python rag_retrieval_eval.py --dataset zh_int --noise_rate {rate} --num 5"
    os.system(cmd)
```

> 可将 `zh_int` 替换为 `zh_refine` 以测试不同数据集。


### 2. 一次性上传模式

- **测试命令：**

```bash
python rag_retrieval_eval.py --dataset zh_refine --all_in_one_upload
python rag_retrieval_eval.py --dataset zh_int --all_in_one_upload
```

> 分别测试 `zh_refine` 和 `zh_int` 数据集的一次性上传检索效果。


## 生成性能测试操作说明

### 1. 逐步上传模式

- **测试脚本：**
    - `test_deepseek.py`
    - `test_Qwen3.py`
- **操作方式：**
    - 分别运行上述脚本，测试不同模型的生成性能。


### 2. 一次性上传模式

- **准备数据：**
    - 先上传合并后的文档文件：
        - `merged_all_docs.txt1`（对应 `zh_refine`）
        - `merged_all_docs.txt2`（对应 `zh_int`）
- **运行脚本：**
    - 执行 `auto.py`，并根据需要修改参数以分别测试不同数据集。

```bash
python auto.py
```


## 常见问题与建议

- 若遇到依赖包缺失，请根据错误提示补充安装。
- 建议在虚拟环境下运行，避免依赖冲突。
- 必须在打开deepseekMine运行的同时才能进行测试
```

如有其他问题，请联系我

