# RAGEval Forge

RAGEval Forge 是一个纯 Node.js 的端到端 RAG 测评工作台，用来把同一份问答数据集跑在不同知识库/问答平台上，并用同一套规则统计准确率。

当前主线只依赖 Node.js，不依赖 Python。

## 能做什么

- 读取 `data/*.json` JSONL 问答数据集。
- 调用 DeepLocals 的真实检索链路：上传 corpus，再调用 `/api/search`。
- 调用 Cherry Studio 的公开本地 API：检索知识库，再调用对话模型回答。
- DeepLocals 和 Cherry Studio 可并发横评。
- 使用同一套答案判定：规则优先，规则无法命中时再走模型裁判。
- 输出 JSON/CSV 报告到 `result-zh/`。
- 提供 Node Web UI，方便本机或另一台 Mac 操作测评。
- 支持从纯文本/Markdown 原文生成 QA 数据集。

## 环境要求

需要 Node.js 20 或更高版本。

```bash
node -v
npm test
```

这个项目目前没有第三方 npm 依赖，使用 Node 内置能力运行。

## 目录结构

```text
.
├─ bin/
│  ├─ rageval.js       # Node CLI 主入口
│  └─ rageval-ui.js    # Node Web UI 启动入口
├─ src/
│  ├─ evaluation/      # 数据集、评分、DeepLocals、Cherry、数据集生成
│  ├─ compare.js       # 并发横评
│  ├─ llm.js           # OpenAI-compatible / Ollama 调用
│  ├─ server.js        # Node Web UI HTTP 服务
│  └─ config.js        # 路径和默认模型配置
├─ test/               # Node 单元测试
├─ data/               # JSONL 数据集
├─ result-zh/          # 测评报告输出
├─ temp_docs/          # 临时 corpus / 运行中间文件
├─ package.json
└─ README.md
```

## 启动 Web UI

```bash
npm start
```

默认地址：

```text
http://127.0.0.1:7861/
```

换端口：

```bash
node bin/rageval-ui.js --host 127.0.0.1 --port 7862
```

## 命令行测评

### DeepLocals

```bash
node bin/rageval.js deeplocals \
  --dataset zh_int_clean \
  --api-base http://127.0.0.1:3335 \
  --limit 1
```

DeepLocals 评测会走真实链路：

1. 找到 `data/<dataset>.corpus.md` 或 `data/<dataset>.corpus.txt`。
2. 如果没有 corpus，就从 gold evidence 生成临时 corpus。
3. 上传 corpus 到 DeepLocals。
4. 对每个问题调用 `/api/search`。
5. 把返回的 `result_prompt` 发给回答模型。
6. 用规则 + 模型裁判计算准确率。

### Cherry Studio

Cherry 需要先在 Cherry Studio 里创建知识库并上传同一份原始文档，然后开启 Cherry API 服务。

```bash
export CHERRY_API_KEY="cs-sk-..."

node bin/rageval.js cherry \
  --dataset zh_int_clean \
  --api-base http://127.0.0.1:23333 \
  --knowledge-base-id <Cherry知识库ID> \
  --model silicon:deepseek-ai/DeepSeek-V4-Flash \
  --document-count 20 \
  --limit 1
```

### 并发横评

```bash
export SILICONFLOW_API_KEY="sk-..."
export CHERRY_API_KEY="cs-sk-..."

node bin/rageval.js compare \
  --dataset zh_int_clean \
  --deepseekmine-api-base http://127.0.0.1:3335 \
  --cherry-api-base http://127.0.0.1:23333 \
  --cherry-knowledge-base-id <Cherry知识库ID> \
  --cherry-model silicon:deepseek-ai/DeepSeek-V4-Flash \
  --deeplocal-answer-provider siliconflow \
  --deeplocal-answer-model deepseek-ai/DeepSeek-V4-Flash \
  --document-count 20
```

## 数据集格式

`data/<dataset>.json` 是 JSONL，每行一条 QA：

```json
{
  "id": 0,
  "query": "问题",
  "answer": ["标准答案"],
  "evidence": [
    {
      "source_file": "来源文件",
      "page": "",
      "quote": "能直接支持答案的原文证据"
    }
  ],
  "positive": ["同 evidence.quote，用于兼容旧数据"]
}
```

## 从文本生成数据集

Web UI 里可以粘贴纯文本/Markdown 原文，生成 JSONL 数据集和 corpus。

命令行入口目前以 UI 为主；核心实现位于：

```text
src/evaluation/datasetBuilder.js
```

生成数据集需要可用的回答模型配置，例如：

```bash
export MIMO_API_KEY="..."
```

或：

```bash
export SILICONFLOW_API_KEY="..."
```

## 密钥配置

不要把真实密钥写进仓库。建议使用环境变量：

```bash
export MIMO_API_KEY="..."
export SILICONFLOW_API_KEY="..."
export CHERRY_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

本地私密文件也会被 `.gitignore` 忽略：

```text
.mimo_api_key
.mineru_api_key
.rageval_api_config.json
.rageval_ollama_context.json
```

## 输出报告

报告写入：

```text
result-zh/
```

主要指标：

- `qa_accuracy`
- `qa_correct`
- `qa_total`
- `evidence_hit_rate`
- `evidence_recall`
- `mrr`
- `avg_retrieved_count`

## 纯 Node 状态

当前仓库主线已经是纯 Node.js：

- 没有 Python 运行入口。
- 没有 Python 依赖清单。
- Web UI、CLI、评测、评分、报告输出都在 Node 中实现。
- 原 Python 主线代码已移除。

需要外部准备的仍然是被测平台本身：DeepLocals、Cherry Studio、NotebookLM、腾讯 ima 等。它们不是本仓库的一部分。
