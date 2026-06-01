import path from "node:path";
import { RESULT_ZH_DIR } from "../config.js";
import { requestJson } from "../http.js";
import { ensureDir, isoTimestamp, writeJson } from "../utils/files.js";
import { answersFor, evidenceQuotesFor, filterRowsByIds, loadJsonl, parseIds, resolveDatasetPath } from "./dataset.js";
import { scoreAnswerWithFallback } from "./judge.js";
import { isAnswerCorrect, scoreRetrievalFromTexts, summarizeFlatQa } from "./scoring.js";

export function cherryDefaults() {
  return {
    datasetName: "",
    datasetPath: "",
    apiBase: "http://127.0.0.1:23333",
    apiKey: "",
    knowledgeBaseIds: "",
    model: "",
    documentCount: 5,
    limit: 0,
    ids: "",
    output: "",
    timeout: 300,
    judgeMode: "rule_then_model",
    temperature: 0.0,
    maxTokens: 1024,
    judgeConfig: null
  };
}

function readCherryKey(rawKey = "") {
  const key = String(rawKey || process.env.CHERRY_API_KEY || "").trim();
  if (!key) throw new Error("Cherry API key is missing. Set CHERRY_API_KEY or pass --api-key.");
  return key;
}

function headers(apiKey) {
  return {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json"
  };
}

async function cherryJson(method, apiBase, apiKey, route, payload = null, timeout = 60) {
  const data = await requestJson(`${apiBase.replace(/\/+$/u, "")}${route}`, {
    method,
    headers: headers(apiKey),
    body: payload ? JSON.stringify(payload) : null,
    timeout
  });
  if (data?.error && typeof data.error === "object") {
    throw new Error(`Cherry API error: ${data.error.message ?? JSON.stringify(data.error)}`);
  }
  return data;
}

function splitIds(value) {
  return String(value || "")
    .split(/[,\s]+/u)
    .map((part) => part.trim())
    .filter(Boolean);
}

async function listKnowledgeBases(apiBase, apiKey, timeout) {
  const data = await cherryJson("GET", apiBase, apiKey, "/v1/knowledge-bases?limit=100", null, timeout);
  const bases = data.knowledge_bases ?? data.data ?? [];
  return Array.isArray(bases) ? bases : [];
}

async function listModels(apiBase, apiKey, timeout) {
  const data = await cherryJson("GET", apiBase, apiKey, "/v1/models?limit=100", null, timeout);
  return Array.isArray(data.data) ? data.data : [];
}

function chooseDefaultModel(models) {
  const blocked = /embedding|rerank|bge/iu;
  const preferred = /qwen3-8b|deepseek-v3|qwen2\.5-7b/iu;
  const candidates = models.map((item) => String(item?.id ?? "").trim()).filter(Boolean);
  const preferredModel = candidates.find((model) => preferred.test(model) && !blocked.test(model));
  if (preferredModel) return preferredModel;
  const firstChatModel = candidates.find((model) => !blocked.test(model));
  if (firstChatModel) return firstChatModel;
  throw new Error("No usable chat model found from Cherry /v1/models.");
}

async function searchKnowledge(apiBase, apiKey, query, knowledgeBaseIds, documentCount, timeout) {
  const payload = {
    query,
    document_count: Math.max(1, Math.min(Number.parseInt(String(documentCount), 10) || 1, 20))
  };
  if (knowledgeBaseIds.length) payload.knowledge_base_ids = knowledgeBaseIds;
  return cherryJson("POST", apiBase, apiKey, "/v1/knowledge-bases/search", payload, timeout);
}

function buildAnswerPrompt(query, results) {
  const chunks = [];
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    const content = String(result?.pageContent ?? "").trim();
    if (!content) continue;
    const metadata = result?.metadata && typeof result.metadata === "object" ? result.metadata : {};
    const sourceText = String(metadata.source ?? metadata.file ?? metadata.path ?? "").trim();
    const kbName = String(result?.knowledge_base_name ?? "").trim();
    const title = [kbName, sourceText].filter(Boolean).join(" / ");
    const heading = title ? `[${index + 1}] ${title}` : `[${index + 1}]`;
    chunks.push(`${heading}\n${content}`);
  }
  const context = chunks.join("\n\n").trim() || "未检索到可用资料。";
  return `
请只根据下面的资料回答问题。要求：
1. 答案尽量简短，直接给出结论。
2. 如果资料不足以回答，请回答“资料不足”。
3. 不要使用资料以外的常识补全。

资料：
${context}

问题：${query}
`.trim();
}

async function cherryChatCompletion(apiBase, apiKey, model, prompt, timeout, maxTokens, temperature) {
  const payload = {
    model,
    messages: [
      { role: "system", content: "你是一个严谨的中文问答助手。只根据用户提供的资料回答。" },
      { role: "user", content: prompt }
    ],
    temperature,
    max_tokens: maxTokens,
    stream: false
  };
  try {
    const data = await cherryJson("POST", apiBase, apiKey, "/v1/chat/completions", payload, timeout);
    const message = data?.choices?.[0]?.message;
    if (!message || typeof message !== "object") throw new Error(`Cherry chat response missing message: ${JSON.stringify(data).slice(0, 1000)}`);
    return String(message.content ?? message.reasoning_content ?? "").trim();
  } catch (error) {
    if (!String(error.message).includes("does not support Chat Completions API")) throw error;
  }
  const messagesPayload = {
    model,
    system: "你是一个严谨的中文问答助手。只根据用户提供的资料回答。",
    messages: [{ role: "user", content: prompt }],
    temperature,
    max_tokens: maxTokens,
    stream: false
  };
  const data = await cherryJson("POST", apiBase, apiKey, "/v1/messages", messagesPayload, timeout);
  if (typeof data.content === "string") return data.content.trim();
  if (!Array.isArray(data.content)) throw new Error(`Cherry messages response missing content: ${JSON.stringify(data).slice(0, 1000)}`);
  const parts = data.content
    .filter((item) => item && typeof item === "object" && item.type === "text" && String(item.text ?? "").trim())
    .map((item) => String(item.text).trim());
  if (parts.length) return parts.join("\n").trim();
  throw new Error(`Cherry messages response has no text content: ${JSON.stringify(data).slice(0, 1000)}`);
}

async function scoreAnswer(query, answers, prediction, goldQuotes, judgeMode, timeout, judgeConfig) {
  if (judgeMode === "rule") {
    return { correct: isAnswerCorrect(prediction, answers), method: "rule", judge: null };
  }
  return scoreAnswerWithFallback(query, answers, prediction, goldQuotes, timeout, judgeConfig);
}

export async function runCherryEval(rawOptions, progress = () => {}) {
  const options = { ...cherryDefaults(), ...rawOptions };
  if (!options.datasetName) throw new Error("--dataset is required");
  const apiKey = readCherryKey(options.apiKey);
  const datasetPath = await resolveDatasetPath(options.datasetName, options.datasetPath);
  const rows = filterRowsByIds(await loadJsonl(datasetPath, Number(options.limit || 0)), parseIds(options.ids));
  if (!rows.length) throw new Error("Dataset has no rows");

  const bases = await listKnowledgeBases(options.apiBase, apiKey, Math.min(options.timeout, 60));
  let knowledgeBaseIds = splitIds(options.knowledgeBaseIds);
  if (!knowledgeBaseIds.length) {
    if (bases.length === 1 && String(bases[0]?.id ?? "").trim()) knowledgeBaseIds = [String(bases[0].id).trim()];
    else throw new Error("Pass --knowledge-base-id when Cherry has zero or multiple knowledge bases.");
  }
  const models = await listModels(options.apiBase, apiKey, Math.min(options.timeout, 60));
  const model = String(options.model || "").trim() || chooseDefaultModel(models);
  const knowledgeNames = Object.fromEntries(
    bases
      .filter((base) => String(base?.id ?? "").trim())
      .map((base) => [String(base.id), String(base.name ?? "")])
  );

  const results = [];
  for (let index = 0; index < rows.length; index += 1) {
    const row = rows[index];
    const query = String(row.query ?? "").trim();
    const answers = answersFor(row);
    const goldQuotes = evidenceQuotesFor(row);
    progress(`[${index + 1}/${rows.length}] Cherry 搜索：${query.slice(0, 80)}`);
    const item = { id: row.id, query, answer: answers, gold_evidence: goldQuotes };
    try {
      const searchPayload = await searchKnowledge(
        options.apiBase,
        apiKey,
        query,
        knowledgeBaseIds,
        options.documentCount,
        options.timeout
      );
      const searchResults = Array.isArray(searchPayload.results) ? searchPayload.results : [];
      const retrieval = scoreRetrievalFromTexts(searchResults.map((result) => String(result?.pageContent ?? "")), goldQuotes, answers);
      const prompt = buildAnswerPrompt(query, searchResults);
      const prediction = await cherryChatCompletion(
        options.apiBase,
        apiKey,
        model,
        prompt,
        options.timeout,
        options.maxTokens,
        options.temperature
      );
      const scored = await scoreAnswer(query, answers, prediction, goldQuotes, options.judgeMode, options.timeout, options.judgeConfig);
      Object.assign(item, {
        retrieval,
        prediction,
        answer_correct: scored.correct,
        answer_judge_method: scored.method,
        search_results: searchResults,
        prompt_chars: prompt.length
      });
      if (scored.judge) item.answer_judge = scored.judge;
      progress(`[${index + 1}/${rows.length}] 判定：${scored.correct ? "正确" : "未命中"}；检索 ${retrieval.retrieved_count} 条`);
    } catch (error) {
      Object.assign(item, {
        retrieval: { evidence_hit: false, evidence_recall: 0.0, mrr: 0.0, retrieved_count: 0 },
        prediction: "",
        answer_correct: false,
        answer_judge_method: "error",
        answer_judge: { correct: false, reason: error.message }
      });
      progress(`[${index + 1}/${rows.length}] 错误：${error.message.slice(0, 160)}`);
    }
    results.push(item);
  }

  const report = {
    schema: "cherry-rag-e2e-node-v1",
    created_at: isoTimestamp(),
    dataset_name: options.datasetName,
    dataset_path: datasetPath,
    api_base: options.apiBase,
    knowledge_base_ids: knowledgeBaseIds,
    knowledge_bases: knowledgeBaseIds.map((id) => ({ id, name: knowledgeNames[id] ?? "" })),
    model,
    document_count: options.documentCount,
    judge_mode: options.judgeMode,
    summary: summarizeFlatQa(results),
    stopped_reason: "",
    completed: results.length === rows.length,
    results
  };
  const output = options.output || path.join(RESULT_ZH_DIR, `cherry_qa_${options.datasetName}_${Math.floor(Date.now() / 1000)}.json`);
  await ensureDir(path.dirname(output));
  report.output_path = output;
  await writeJson(output, report);
  return report;
}
