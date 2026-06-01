# RAGEval Forge

RAGEval Forge 是一个本地 RAG 测评工作台，用来完成两件事：

1. 从 PDF、Office 文档、图片或文本资料生成可追溯的问答测评集。
2. 把生成好的测评集接入 `deepseekmine` 的真实 RAG 链路，测试检索、Prompt 组装和最终回答效果。

项目主目录：

```text
E:\RAG
```

本 README 只描述 `E:\RAG` 内的真实代码。当前 Codex 工作目录可能显示在 OneDrive 目录下，但实际 UI、生成逻辑、测评逻辑都在 `E:\RAG`。

## 一句话流程

```text
上传 PDF/文档
  -> MinerU 解析为 Markdown + JSON 语义资产
  -> 按标题层级/证据块生成 QA + evidence
  -> 保存到 E:\RAG\data
  -> 选择整个数据集
  -> 上传对应语料到 deepseekmine 新知识库
  -> 对每个问题调用 deepseekmine /api/search
  -> 用 result_prompt 调 MiMo 生成答案
  -> 规则判定 + MiMo 兜底裁判
  -> 输出真实 RAG 测评报告
```

## 目录结构

```text
E:\RAG
├─ rageval                        # 当前主线代码包
│  ├─ ui                          # Web UI，默认端口 7861
│  ├─ builder                     # 从文档/Markdown 生成 QA 数据集
│  ├─ mineru                      # MinerU 解析、PDF 分页、结果下载合并
│  ├─ deepseekmine                # deepseekmine 分块/sidecar 的兼容上传辅助
│  ├─ evaluation                  # 调 deepseekmine 做真实 RAG 测评
│  ├─ datasets                    # 数据集相关辅助预留
│  └─ config.py                   # 本地路径、模型与密钥文件配置
├─ scripts                        # 新命令行入口
│  ├─ run_ui.py
│  ├─ build_dataset.py
│  └─ eval_deepseekmine.py
├─ legacy                         # 旧实验脚本原样归档
├─ tests                          # 测试目录
├─ rag_dataset_ui.py              # 兼容入口，转发到 rageval.ui.app
├─ rag_dataset_builder.py         # 兼容入口，转发到 rageval.builder.dataset_builder
├─ mineru_ocr.py                  # 兼容入口，转发到 rageval.mineru.ocr
├─ deepseekmine_rag_eval.py       # 兼容入口，转发到 rageval.evaluation.deepseekmine_rag_eval
├─ deepseekmine_compat_import.py  # 兼容入口，转发到 rageval.deepseekmine.compat_import
├─ data                           # 数据集、corpus、MinerU 资产、缓存
├─ result-zh                      # deepseekmine 真实测评报告输出目录
├─ temp_docs                      # UI 临时上传、运行日志
├─ logs                           # 旧脚本或辅助日志
├─ .mimo_api_key                  # MiMo key，本地私密文件，不应提交
├─ .mineru_api_key                # MinerU token，本地私密文件，不应提交
├─ merged_all_docs2.txt           # zh_int_clean 当前真实原始语料
└─ requirements.txt               # 旧依赖清单，当前功能还需要 requests、pypdf/PyPDF2 等
```

## 快速启动 UI

推荐在 PowerShell 中启动：

```powershell
cd E:\RAG
python E:\RAG\rag_dataset_ui.py --host 127.0.0.1 --port 7861
```

浏览器打开：

```text
http://127.0.0.1:7861/
```

如果端口被占用，可以换端口：

```powershell
python E:\RAG\rag_dataset_ui.py --host 127.0.0.1 --port 7862
```

## 密钥配置

项目会优先读取环境变量，其次读取本地文件。

MiMo：

```text
环境变量：MIMO_API_KEY
本地文件：E:\RAG\.mimo_api_key
Base URL：https://token-plan-cn.xiaomimimo.com/v1
模型：mimo-v2.5-pro
```

对话模型也可以在 UI 的 `API Settings / 对话模型 API 配置` 中选择和保存。配置文件为：

```text
E:\RAG\.rageval_api_config.json
```

当前预置支持：

- MiMo：`mimo-v2.5-pro`
- 硅基流动：`Qwen/Qwen3.5-4B`，默认 `https://api.siliconflow.cn/v1`
- DeepSeek：`deepseek-chat`
- 阿里云百炼：`qwen-plus`
- 自定义 OpenAI-compatible API

硅基流动可使用环境变量：

```text
SILICONFLOW_API_KEY
```

MinerU：

```text
环境变量：MINERU_API_TOKEN
本地文件：E:\RAG\.mineru_api_key
API Base：https://mineru.net/api/v4
```

不要把真实密钥写进 README、报告或 Git 提交里。

## UI 里的两个主功能

### 1. 生成问答对与证据

位置：页面左侧或上半部分的 `生成问答对与证据`。

输入：

- PDF、Word、PPT、Excel、图片、HTML 等 MinerU 支持的文件。
- 也支持文本、Markdown、JSONL、CSV 等本地解析文件。

输出：

```text
E:\RAG\data\<dataset>.json
E:\RAG\data\<dataset>.json.report.json
E:\RAG\data\<dataset>.corpus.md
E:\RAG\data\assets\<dataset>\...
```

生成的数据集是 JSONL，每一行是一条 QA：

```json
{
  "id": 0,
  "query": "问题",
  "answer": ["标准答案"],
  "evidence": [
    {
      "source_file": "来源文件",
      "page": "页码或位置",
      "quote": "能够直接支持答案的原文证据"
    }
  ],
  "positive": ["同 evidence.quote，用于兼容旧评测脚本"]
}
```

业务测评不需要正负例。现在负例默认不生成，真实评测只使用：

- `query`
- `answer`
- `evidence`
- `corpus` 或 MinerU 资产

### 2. 真实 RAG 测评

位置：页面里的 `REAL RAG EVALUATION / 选择整个数据集进行真实测评`。

这个功能不是测模型自己的 RAG 能力，而是模拟真实业务链路：

1. 选择一个完整数据集。
2. 每次新建一个 deepseekmine 知识库。
3. 上传该数据集对应的原始语料或 MinerU Markdown + sidecar。
4. 对数据集里的全部问题逐条调用 deepseekmine 搜索接口。
5. 取 deepseekmine 返回的 `result_prompt`。
6. 把完整 `result_prompt` 发给 MiMo 回答。
7. 用标准答案做判定，规则无法稳定判断时再让 MiMo 做裁判兜底。

测评报告输出到：

```text
E:\RAG\result-zh\deepseekmine_<mode>_<dataset>_<timestamp>.json
```

主要指标：

- `evidence_hit_rate`：deepseekmine 检索结果是否命中标准证据。
- `evidence_recall`：标准证据覆盖情况。
- `mrr`：首个命中证据的排序质量。
- `qa_accuracy`：最终回答是否匹配标准答案。

## deepseekmine 是否会自动启动

不会。

运行真实测评前，需要你自己启动 `E:\deepseekmine`，并确认 UI 中填写的地址可以访问，默认是：

```text
http://127.0.0.1:3335
```

如果 deepseekmine 没启动，UI 会给出友好提示：

```text
无法连接 deepseekmine 服务：http://127.0.0.1:3335。
请先启动 E:\deepseekmine 项目，确认地址能打开后再运行真实测评。
```

## MinerU 解析与大 PDF

MinerU 接入代码在：

```text
E:\RAG\mineru_ocr.py
```

调用流程：

```text
POST /api/v4/file-urls/batch
PUT 上传文件到 MinerU 返回的 upload URL
GET /api/v4/extract-results/batch/{batch_id}
下载 full_zip_url
解压 Markdown、content list、middle json 等资产
```

PDF 超过 200 页时，系统会自动分页：

```text
原 PDF
  -> part1 1-200
  -> part2 201-400
  -> part3 401-...
  -> 分别送 MinerU
  -> 按原页序合并 Markdown
```

生成后会保存：

```text
E:\RAG\data\assets\<dataset>\originals
E:\RAG\data\assets\<dataset>\mineru
E:\RAG\data\assets\<dataset>\manifest.json
E:\RAG\data\mineru_cache
```

这些资产用于后续复用。也就是说同一个 PDF 后续再次生成 QA 时，可以复用之前的 MinerU Markdown/JSON，不必重复 OCR。

## deepseekmine 分块对齐

新生成的 PDF/MinerU 数据集会尽量对齐 deepseekmine 的真实分块：

- 保存 MinerU Markdown。
- 保存 MinerU 返回的 JSON 语义资产。
- 写出 deepseekmine 兼容 sidecar。
- 真实测评时通过 deepseekmine 原生上传接口导入 Markdown + sidecar。

相关代码：

```text
E:\RAG\deepseekmine_compat_import.py
E:\RAG\deepseekmine_rag_eval.py
```

以前为了测试曾在 `E:\deepseekmine` 里加过桥接接口；现在已经不依赖那部分新增文件。当前策略是尽量只改 `E:\RAG` 侧，通过 deepseekmine 现有上传能力完成测评。

## 老数据集与原始语料

`E:\RAG\data` 里有一些老数据集，不是通过当前 UI/MinerU 流程生成的。它们可能缺少 `corpus` 或 MinerU 资产。

当前重点老数据集：

```text
数据集：E:\RAG\data\zh_int_clean.json
题数：100
真实原始语料：E:\RAG\merged_all_docs2.txt
```

`zh_int_clean` 的报告里已经记录原始语料路径。真实测评时会直接上传：

```text
E:\RAG\merged_all_docs2.txt
```

不会再上传复制出来的 `.md` 或 `.txt` corpus。

## 数据集列表状态含义

UI 中每个数据集会显示状态：

```text
deepseekmine 对齐可测
```

表示它有 corpus，也有 MinerU 资产或 sidecar，适合走最接近 deepseekmine PDF 解析后的真实链路。

```text
文本语料可测
```

表示它有可上传的文本语料，例如 `.txt` 或 `.corpus.md`，可以跑真实测评，但不是 MinerU 资产级对齐。

```text
旧格式，缺 corpus，需用本 UI 重新生成
```

表示只有旧 JSON 问答，没有对应语料。可以作为历史记录看，但不能直接真实测评。

## 命令行生成数据集

UI 是推荐入口。命令行也可以直接跑：

```powershell
python E:\RAG\rag_dataset_builder.py `
  --input E:\docs\my_report.pdf `
  --output E:\RAG\data\my_eval.json `
  --backend mimo `
  --model mimo-v2.5-pro `
  --target-count 100 `
  --items-per-chunk 1 `
  --negative-count 0
```

常用参数：

- `--input`：输入文件、目录或 zip。
- `--output`：输出 JSONL。
- `--target-count`：目标问题数。
- `--max-per-source`：每个源文件最多生成多少题。
- `--items-per-chunk`：每块最多生成多少题，UI 固定为 1。
- `--backend`：`mimo`、`openai`、`ollama`、`none`。
- `--allow-heuristic`：只适合冒烟测试，不适合正式测评集。

## 命令行真实测评

deepseekmine 已启动后，可以这样跑：

```powershell
python E:\RAG\deepseekmine_rag_eval.py `
  --dataset zh_int_clean `
  --api-base http://127.0.0.1:3335 `
  --mode qa `
  --timeout 900
```

只测检索，不让 MiMo 回答：

```powershell
python E:\RAG\deepseekmine_rag_eval.py `
  --dataset zh_int_clean `
  --api-base http://127.0.0.1:3335 `
  --mode retrieval
```

指定外部语料：

```powershell
python E:\RAG\deepseekmine_rag_eval.py `
  --dataset zh_int_clean `
  --corpus-path E:\RAG\merged_all_docs2.txt `
  --api-base http://127.0.0.1:3335 `
  --mode qa
```

## 进度、中断与失败处理

UI 中生成和测评都会显示进度日志：

- MinerU 上传/轮询/下载阶段。
- Markdown 或语料解析阶段。
- QA 分块阶段。
- QA 逐块生成阶段。
- deepseekmine 上传阶段。
- 每道题检索与答案判定阶段。

长任务支持中断。中断后已写出的报告会尽量保留当前进度，并记录 `stopped_reason`。

MiMo API 常见情况：

- `402`：余额不足。
- `429`：请求过频或 Token Plan 额度耗尽。
- `500/503`：服务侧临时错误。

当前测评逻辑会对部分服务错误重试。遇到额度不足或请求超限时，会保护性中止并写入已完成部分，不会让整份报告完全丢失。

## 删除数据集

UI 数据集列表里有删除按钮。删除会清理：

- `E:\RAG\data\<dataset>.json`
- `E:\RAG\data\<dataset>.corpus.txt`
- `E:\RAG\data\<dataset>.corpus.md`
- `E:\RAG\data\<dataset>.corpus.generated.md`
- `E:\RAG\data\<dataset>.json.report.json`
- `E:\RAG\data\assets\<dataset>`
- `E:\RAG\result-zh\deepseekmine_*_<dataset>_*.json`

外部原始语料不会被删，例如：

```text
E:\RAG\merged_all_docs2.txt
```

## 推荐的正式测评做法

1. 对业务 PDF 使用 UI 生成新数据集。
2. 检查 `*.json.report.json`，确认 MinerU 成功、QA 数量达标、拒绝数量合理。
3. 抽查 5 到 10 条 QA，确认：
   - 问题不含歧义。
   - 答案能从 evidence 直接推出。
   - evidence 是原文证据，不是模型总结。
4. 启动 `E:\deepseekmine`。
5. 在 UI 选择整个数据集跑真实测评。
6. 查看 `result-zh` 中的报告，重点看：
   - 未命中的题。
   - 检索命中但回答错的题。
   - 回答对但证据没命中的题。
   - Prompt 长度异常的题。

## 常见问题

### 为什么不需要负例？

真实业务测评是“问题 -> deepseekmine 检索 -> result_prompt -> 回答模型 -> 对标准答案”。核心是标准答案和证据，不需要训练式正负例。

### 为什么每次真实测评都新建知识库？

为了模拟真实用户上传文件后的独立测试，避免旧知识库残留、缓存或污染影响结果。

### 为什么有的数据集显示文本语料可测，而不是 deepseekmine 对齐可测？

因为它不是当前 UI 通过 MinerU 生成的，没有 Markdown + JSON sidecar 资产。它仍然可以上传文本 corpus 测，但无法保证 PDF 解析分块完全一致。

### 旧的 `zh_int_clean` 现在怎么测？

直接在 UI 选择 `zh_int_clean`。它有 100 条 QA，测评时会直接上传：

```text
E:\RAG\merged_all_docs2.txt
```

### deepseekmine 没启动会怎样？

真实测评会在上传前做健康检查。如果连不上，会停止并提示先启动 deepseekmine，不再展示大段 Python traceback。

### 报告下载失败怎么办？

报告保存在本地：

```text
E:\RAG\result-zh
```

即使浏览器下载失败，也可以直接到这个目录打开最新的 `deepseekmine_*.json`。

## 维护提醒

- 不要把 `.mimo_api_key`、`.mineru_api_key` 提交到 Git。
- 不要随意删除 `data/assets/<dataset>`，否则会丢失 MinerU 复用资产。
- 老数据集如果缺 corpus，建议优先补真实原始语料路径，而不是用 evidence 拼临时 corpus。
- 对正式报告，保留对应数据集 JSON、report JSON、原始语料和 deepseekmine 测评报告，方便复现。
