# RAGEval Forge Node 版使用说明

这个仓库现在新增了纯 Node.js 的测评主链路。目标是让另一个用户在 Mac 或 Windows 上不依赖 Python，也能按同一套逻辑测 DeepLocals、Cherry Studio，以及后续接入 NotebookLM / 腾讯 ima 这类 UI 平台。

## 环境

需要 Node.js 20 或更高版本。

```bash
npm test
```

## DeepLocals 单平台测评

```bash
node bin/rageval.js deeplocals \
  --dataset zh_int_clean \
  --api-base http://127.0.0.1:3335 \
  --limit 1
```

如果仓库里没有对应 corpus 文件，Node 版会和 Python 版一样从 gold evidence 生成临时 corpus，再上传到 DeepLocals。

## Cherry Studio 单平台测评

Cherry 需要先在 Cherry Studio 里手动建好知识库并上传同一份文档，然后启动 Cherry API 服务。

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

## DeepLocals vs Cherry 并发横评

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

输出仍然写到：

```text
result-zh/
```

## 与旧 Python 逻辑保持一致的部分

- 数据集仍然是 `data/<dataset>.json` JSONL。
- `answer` 和 `evidence.quote` 的读取方式一致。
- DeepLocals 仍然调用 `/api/files/upload` 和 `/api/search`，不绕过检索链路。
- Cherry 仍然调用 `/v1/knowledge-bases/search` 和模型对话接口。
- 答案评分仍然是规则优先，规则失败后再走二次模型裁判。
- 输出报告包含 `summary.qa_accuracy`、`evidence_hit_rate`、`evidence_recall`、`mrr`、`avg_retrieved_count`。

## 还没有迁移的部分

- MinerU / PDF 解析生成数据集仍然保留 Python 版本。
- Web UI 仍然是 Python UI，下一步可以改成 Node/Electron 或 Node Web UI。
- NotebookLM / 腾讯 ima 的 UI 自动化适配器还没有正式写进 Node CLI。
