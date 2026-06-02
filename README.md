# RAGEval Forge

RAGEval Forge 是一个纯 Node.js 的端到端 RAG 测评工作台。它用于从文档生成可追溯问答数据集，并把同一套问答集放到 DeepLocals、Cherry Studio、NotebookLM、ima 等平台上做横向问答准确率测评。

当前主线已经迁移为纯 Node 项目：仓库内没有 Python 运行入口，也没有 Python 依赖清单。

## 迁移状态

Node 版已经对齐旧 Python 版的核心功能逻辑：

- 上传 PDF、Office、图片、HTML、TXT、Markdown 等原始文档。
- 使用 MinerU OCR/解析文档，PDF 超过 200 页时自动拆分任务再合并结果。
- 从解析后的文档生成 JSONL 问答数据集、corpus 和资产 manifest。
- 生成 DeepLocals 兼容的 MinerU 分块 sidecar。
- DeepLocals 真实测评时优先走原生知识库上传，携带 MinerU Markdown 和 sidecar。
- 调用 DeepLocals `/api/search` 获取检索结果和 `result_prompt`。
- 使用同一对话模型生成最终回答。
- 使用“规则优先，模型二次裁判兜底”的问答评分逻辑。
- 支持 Cherry Studio API 横评，并可和 DeepLocals 并发测评。
- 支持任务进度、取消、日志、报告下载、数据集下拉选择和删除。
- 支持 API 设置面板配置 OpenAI-compatible、MiMo、SiliconFlow、DeepSeek、DashScope、Ollama 等模型服务。

需要注意：Node 版不是逐行复刻 Python 源码，也不保证 LLM 输出逐 token 一致；但产品功能和测评业务链路已经对齐旧 Python 版。最终指标仍会受到外部平台状态、模型随机性、API 配额和文档上传配置影响。

## 环境要求

需要 Node.js 20 或更高版本。

```bash
node -v
npm install
npm test
```

## 启动 Web UI

默认端口是 7861：

```bash
npm start
```

也可以指定端口，例如当前常用的 7871：

```bash
node bin/rageval-ui.js --host 127.0.0.1 --port 7871
```

打开：

```text
http://127.0.0.1:7871/
```

## 命令行入口

生成数据集：

```bash
node bin/rageval.js build \
  --dataset custom_rag_eval \
  --input E:\docs\a.pdf,E:\docs\b.md \
  --target-count 100 \
  --questions-per-chunk 3
```

使用纯文本生成数据集：

```bash
node bin/rageval.js build \
  --dataset custom_rag_eval \
  --text-file E:\docs\source.txt
```

DeepLocals 真实测评：

```bash
node bin/rageval.js deeplocals \
  --dataset zh_int_clean \
  --api-base http://127.0.0.1:3335 \
  --limit 1
```

Cherry Studio 测评：

```bash
node bin/rageval.js cherry \
  --dataset zh_int_clean \
  --api-base http://127.0.0.1:23333 \
  --knowledge-base-id <Cherry知识库ID> \
  --model silicon:deepseek-ai/DeepSeek-V4-Flash \
  --document-count 20 \
  --limit 1
```

DeepLocals 和 Cherry Studio 并发横评：

```bash
node bin/rageval.js compare \
  --dataset zh_int_clean \
  --deeplocal-api-base http://127.0.0.1:3335 \
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
  "positive": ["兼容旧数据的证据文本"]
}
```

## 外部服务

项目本身是纯 Node，但真实测评仍依赖外部平台或服务：

- DeepLocals：本地服务通常是 `http://127.0.0.1:3335`。
- Cherry Studio：需要先在 Cherry 中创建知识库、上传同一份文档，并开启 API 服务。
- MinerU：用于 PDF/Office/图片等文档解析。
- SiliconFlow、MiMo、DeepSeek、DashScope、Ollama 等：用于生成 QA、回答和模型裁判。
- NotebookLM、腾讯 ima 等 UI 平台：需要通过 UI 自动化或人工预上传文档后再测试。

## 密钥配置

不要把真实密钥写进仓库。推荐使用环境变量：

```bash
set SILICONFLOW_API_KEY=...
set CHERRY_API_KEY=...
set MIMO_API_KEY=...
set MINERU_API_TOKEN=...
```

也可以使用本地忽略文件：

```text
.mimo_api_key
.mineru_api_key
.rageval_api_config.json
.rageval_ollama_context.json
```

这些文件已在 `.gitignore` 中忽略。

## 输出文件

测评报告写入：

```text
result-zh/
```

常见指标：

- `qa_accuracy`
- `qa_correct`
- `qa_total`
- `evidence_hit_rate`
- `evidence_recall`
- `mrr`
- `avg_retrieved_count`

数据集生成的文档资产写入：

```text
data/assets/
data/mineru_cache/
```

其中 `data/mineru_cache/` 是本地缓存，默认不提交。当前仓库已随代码发布部分可复现实验数据集及其 `data/assets/` 资产，方便其他机器拉取后直接做测评。

已发布的数据集包括：

- `zh_int`
- `zh_int_clean`
- `zh_refine`
- `100_rag_eval`
- `1000_rag_eval`
- `custom_rag_eval`
- `mineru100_rag_eval`

## 验证命令

发布前建议至少运行：

```bash
npm test
node --check src/server.js
node --check src/evaluation/datasetBuilder.js
node --check src/evaluation/deeplocals.js
node --check src/evaluation/deeplocalsCompat.js
```

当前单元测试覆盖了评分逻辑、JSONL 读取、裁判 JSON 修复、DeepLocals MinerU sidecar 生成等关键路径。
