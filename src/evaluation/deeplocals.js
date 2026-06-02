import fs from "node:fs/promises";
import path from "node:path";
import { DATA_DIR, RESULT_ZH_DIR } from "../config.js";
import { requestJson } from "../http.js";
import { chatCompletion, loadLlmConfig, messageText } from "../llm.js";
import { ensureDir, isoTimestamp, timestampCompact, writeJson } from "../utils/files.js";
import {
  answersFor,
  buildGeneratedCorpus,
  buildSelectedCorpus,
  evidenceQuotesFor,
  filterRowsByIds,
  loadJsonl,
  parseIds,
  resolveDatasetAndCorpus
} from "./dataset.js";
import { assetMarkdownPaths, uploadMarkdownAssetsViaNativeUpload } from "./deeplocalsCompat.js";
import { LLMQuotaExceededError, scoreAnswerWithFallback } from "./judge.js";
import { scoreRetrievalFromTexts, stripThinking, summarizeDeepLocals } from "./scoring.js";

export function deepLocalsDefaults() {
  return {
    datasetName: "",
    datasetPath: "",
    corpusPath: "",
    apiBase: "http://127.0.0.1:3335",
    knowledgeLabel: "",
    mode: "qa",
    limit: 0,
    ids: "",
    output: "",
    includePrompts: true,
    useAdaptiveRag: true,
    timeout: 900,
    answerConfig: null
  };
}

async function healthCheck(apiBase, timeout = 10) {
  const base = apiBase.replace(/\/+$/u, "");
  try {
    const response = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(timeout * 1000) });
    if (response.status < 500) return;
  } catch {
    // Some builds do not expose /api/health.
  }
  const response = await fetch(`${base}/api/search`, {
    method: "OPTIONS",
    signal: AbortSignal.timeout(timeout * 1000)
  }).catch((error) => {
    throw new Error(`无法连接 DeepLocals 服务：${apiBase}。请先启动 DeepLocals，再运行测评。原始错误：${error.message}`);
  });
  if (response.status >= 500) {
    throw new Error(`DeepLocals 服务异常：${apiBase} 返回 HTTP ${response.status}`);
  }
}

async function uploadCorpus(apiBase, knowledgeLabel, corpusPath, timeout = 900) {
  const form = new FormData();
  form.set("knowledgeLabel", knowledgeLabel);
  form.set("waitForProcessing", "true");
  form.set("fastParsing", "true");
  const bytes = await fs.readFile(corpusPath);
  const contentType = path.extname(corpusPath).toLowerCase() === ".txt" ? "text/plain" : "text/markdown";
  form.set("file", new Blob([bytes], { type: contentType }), path.basename(corpusPath));

  const response = await fetch(`${apiBase.replace(/\/+$/u, "")}/api/files/upload`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(timeout * 1000)
  });
  const text = await response.text();
  if (response.status >= 400) {
    throw new Error(`DeepLocals upload failed: HTTP ${response.status} ${text.slice(0, 1000)}`);
  }
  const payload = JSON.parse(text || "{}");
  if (payload.code !== 0) throw new Error(`DeepLocals upload failed: ${JSON.stringify(payload).slice(0, 1000)}`);
  let docIds = Array.isArray(payload.files) ? payload.files.map((item) => String(item).trim()).filter(Boolean) : [];
  if (!docIds.length) docIds = [path.basename(corpusPath)];
  return { docIds, uploadPayload: payload };
}

async function searchDeepLocals(apiBase, query, knowledgeLabel, docIds, useAdaptiveRag, timeout = 180) {
  return requestJson(`${apiBase.replace(/\/+$/u, "")}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      knowledgeLabel,
      reset: true,
      useAdaptiveRag,
      docIds
    }),
    timeout
  });
}

async function answerWithConfiguredModel(prompt, timeout = 300, config = null) {
  const data = await chatCompletion(
    [
      {
        role: "system",
        content: "你是一个严谨的中文问答助手。只根据用户提供的上下文回答，答案尽量简短，直接给出结论。"
      },
      { role: "user", content: prompt }
    ],
    2048,
    timeout,
    "answer",
    config
  );
  return stripThinking(messageText(data));
}

export async function runDeepLocalsEval(rawOptions, progress = () => {}) {
  const options = { ...deepLocalsDefaults(), ...rawOptions };
  if (!options.datasetName) throw new Error("--dataset is required");
  const { dataset, corpus: initialCorpus } = await resolveDatasetAndCorpus(options.datasetName, options.datasetPath, options.corpusPath);
  const rows = filterRowsByIds(await loadJsonl(dataset, Number(options.limit || 0)), parseIds(options.ids));
  if (!rows.length) throw new Error("Dataset has no rows");

  const selectedIds = parseIds(options.ids);
  let corpusPath = initialCorpus;
  let selectedCorpusPath = "";
  if (!options.knowledgeLabel) {
    options.knowledgeLabel = `rag_eval_${path.basename(dataset, ".json")}_${timestampCompact()}`;
  }
  if (selectedIds.length) {
    selectedCorpusPath = await buildSelectedCorpus(rows, path.basename(dataset, ".json"));
    corpusPath = selectedCorpusPath;
    progress(`已按所选 QA 构建独立测试语料：${corpusPath}`);
  } else if (!corpusPath) {
    corpusPath = await buildGeneratedCorpus(rows, path.basename(dataset, ".json"));
    progress(`未找到 corpus 文件，使用 evidence 生成临时语料：${corpusPath}`);
  }

  progress(`检查 DeepLocals 服务：${options.apiBase}`);
  await healthCheck(options.apiBase);
  const localAssetPaths = selectedIds.length ? [] : await assetMarkdownPaths(path.basename(dataset, ".json"));
  let docIds;
  let uploadPayload;
  if (localAssetPaths.length) {
    const requestedKnowledgeLabel = options.knowledgeLabel;
    progress(`创建 DeepLocals 原生知识库，并上传 MinerU Markdown + sidecar：${localAssetPaths.length} 个文件`);
    const nativeUpload = await uploadMarkdownAssetsViaNativeUpload(
      options.apiBase,
      requestedKnowledgeLabel,
      localAssetPaths,
      options.timeout,
      progress
    );
    options.knowledgeLabel = nativeUpload.knowledgeLabel;
    docIds = nativeUpload.docIds;
    uploadPayload = nativeUpload.uploadPayload;
  } else {
    progress(`上传测评语料到 DeepLocals：${path.basename(corpusPath)}`);
    const corpusUpload = await uploadCorpus(options.apiBase, options.knowledgeLabel, corpusPath, options.timeout);
    docIds = corpusUpload.docIds;
    uploadPayload = corpusUpload.uploadPayload;
  }
  progress(`上传完成，docIds：${docIds.join(", ")}`);

  let answerConfig = options.answerConfig;
  if (options.mode === "qa" && !answerConfig) answerConfig = await loadLlmConfig();

  const results = [];
  let stoppedReason = "";
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const query = String(row.query ?? "").trim();
    const answers = answersFor(row);
    const goldQuotes = evidenceQuotesFor(row);
    progress(`[${index + 1}/${rows.length}] 检索：${query.slice(0, 80)}`);
    const searchResult = await searchDeepLocals(
      options.apiBase,
      query,
      options.knowledgeLabel,
      docIds,
      options.useAdaptiveRag,
      options.timeout
    );
    const hits = Array.isArray(searchResult.hits) ? searchResult.hits : [];
    const retrieval = scoreRetrievalFromTexts(hits.map((hit) => String(hit?.content ?? "")), goldQuotes, answers);
    const item = {
      id: row.id,
      query,
      answer: answers,
      gold_evidence: goldQuotes,
      retrieval,
      hit_titles: hits.slice(0, 10).map((hit) => String(hit?.title ?? "")).filter(Boolean),
      prompt_chars: String(searchResult.result_prompt ?? "").length
    };
    if (options.includePrompts) item.result_prompt = searchResult.result_prompt ?? "";

    if (options.mode === "qa") {
      try {
        const prediction = await answerWithConfiguredModel(String(searchResult.result_prompt ?? ""), options.timeout, answerConfig);
        item.prediction = prediction;
        const scored = await scoreAnswerWithFallback(query, answers, prediction, goldQuotes, options.timeout, answerConfig);
        item.answer_correct = scored.correct;
        item.answer_judge_method = scored.method;
        if (scored.judge) item.answer_judge = scored.judge;
        progress(`[${index + 1}/${rows.length}] 答案判定：${scored.correct ? "正确" : "未命中"}`);
      } catch (error) {
        if (error instanceof LLMQuotaExceededError) {
          stoppedReason = error.message;
          item.prediction = "";
          item.answer_correct = false;
          item.answer_judge_method = "api_quota_exhausted";
          item.answer_judge = { correct: false, reason: stoppedReason };
          results.push(item);
          break;
        }
        item.prediction = "";
        item.answer_correct = false;
        item.answer_judge_method = "mimo_answer_error";
        item.answer_judge = { correct: false, reason: error.message };
        progress(`[${index + 1}/${rows.length}] 回答模型失败，记为未命中并继续：${error.message.slice(0, 160)}`);
      }
    }
    results.push(item);
  }

  const answerBackend = answerConfig?.provider_id ?? null;
  const answerModel = answerConfig?.model ?? null;
  const report = {
    schema: "deeplocals-rag-e2e-node-v1",
    created_at: isoTimestamp(),
    dataset_name: options.datasetName,
    dataset_path: dataset,
    corpus_path: corpusPath,
    uploaded_asset_markdown_paths: localAssetPaths,
    api_base: options.apiBase,
    knowledge_label: options.knowledgeLabel,
    doc_ids: docIds,
    mode: options.mode,
    selected_ids: selectedIds,
    answer_backend: options.mode === "qa" ? answerBackend : null,
    answer_model: options.mode === "qa" ? answerModel : null,
    answer_judge: options.mode === "qa" ? "rule_then_model_fallback" : null,
    upload: uploadPayload,
    summary: summarizeDeepLocals(results),
    stopped_reason: stoppedReason,
    completed: !stoppedReason && results.length === rows.length,
    results
  };
  if (selectedCorpusPath) report.selected_corpus_path = selectedCorpusPath;

  const output = options.output || path.join(RESULT_ZH_DIR, `deeplocals_${options.mode}_${options.datasetName}_${Math.floor(Date.now() / 1000)}.json`);
  await ensureDir(path.dirname(output));
  report.output_path = output;
  await writeJson(output, report);
  return report;
}
