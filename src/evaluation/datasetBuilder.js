import fs from "node:fs/promises";
import path from "node:path";
import { DATA_DIR } from "../config.js";
import { chatCompletion, loadLlmConfig, messageText } from "../llm.js";
import { ensureDir, writeJson } from "../utils/files.js";
import { extractJsonObject } from "./judge.js";
import { stripThinking } from "./scoring.js";

export function sanitizeDatasetName(value) {
  const cleaned = String(value || "").trim().replace(/[^A-Za-z0-9_.-]+/gu, "_");
  return cleaned || "custom_rag_eval";
}

function compactWhitespace(text) {
  return String(text || "").replace(/\r\n/gu, "\n").replace(/[ \t]+/gu, " ").replace(/\n{3,}/gu, "\n\n").trim();
}

export function splitCorpus(text, { maxChars = 1800, overlapChars = 180 } = {}) {
  const normalized = compactWhitespace(text);
  if (!normalized) return [];
  const paragraphs = normalized.split(/\n\s*\n/gu).map((part) => part.trim()).filter(Boolean);
  const chunks = [];
  let current = "";
  for (const paragraph of paragraphs) {
    if (!current) {
      current = paragraph;
      continue;
    }
    if (current.length + paragraph.length + 2 <= maxChars) {
      current += `\n\n${paragraph}`;
    } else {
      chunks.push(current);
      const tail = current.slice(Math.max(0, current.length - overlapChars));
      current = tail ? `${tail}\n\n${paragraph}` : paragraph;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

function parseJsonArray(rawText) {
  const text = stripThinking(rawText).trim().replace(/^```(?:json)?\s*/iu, "").replace(/\s*```$/u, "");
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
    if (Array.isArray(parsed.items)) return parsed.items;
    if (Array.isArray(parsed.questions)) return parsed.questions;
  } catch {
    // Continue with balanced extraction.
  }
  try {
    const obj = extractJsonObject(text);
    if (Array.isArray(obj.items)) return obj.items;
    if (Array.isArray(obj.questions)) return obj.questions;
    if (Array.isArray(obj.qa)) return obj.qa;
  } catch {
    // Continue with bracket extraction.
  }
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start >= 0 && end > start) {
    const repaired = text.slice(start, end + 1).replace(/,\s*([}\]])/gu, "$1");
    const parsed = JSON.parse(repaired);
    if (Array.isArray(parsed)) return parsed;
  }
  throw new Error("LLM did not return a JSON array of QA items.");
}

async function generateQaForChunk(chunk, count, timeout, config) {
  const prompt = `
你是 RAG 评测数据集生成器。请只根据给定原文生成 ${count} 条中文问答。

要求：
1. 问题必须能由原文直接回答。
2. 答案要短，保留关键实体、数字、日期、地点、政策名或列表项。
3. evidence_quote 必须逐字摘自原文，且能支撑答案。
4. 不要编造原文没有的信息。
5. 只输出 JSON 数组，不要 markdown。

JSON 格式：
[
  {"query": "问题", "answer": ["标准答案"], "evidence_quote": "原文证据"}
]

原文：
${chunk}
`.trim();
  const data = await chatCompletion(
    [
      { role: "system", content: "你生成严格、可追溯的 RAG QA 测评数据，只输出 JSON。" },
      { role: "user", content: prompt }
    ],
    1800,
    timeout,
    "dataset generation",
    config
  );
  return parseJsonArray(messageText(data));
}

export async function buildDatasetFromText(options, progress = () => {}) {
  const datasetName = sanitizeDatasetName(options.datasetName);
  const sourceName = String(options.sourceName || `${datasetName}.txt`).trim() || `${datasetName}.txt`;
  const text = compactWhitespace(options.text);
  if (!text) throw new Error("Source text is empty.");

  const targetQuestions = Math.max(1, Number.parseInt(String(options.targetQuestions || 20), 10));
  const questionsPerChunk = Math.max(1, Math.min(Number.parseInt(String(options.questionsPerChunk || 3), 10), 8));
  const timeout = Number.parseInt(String(options.timeout || 300), 10) || 300;
  const config = options.llmConfig || (await loadLlmConfig());
  if (config.provider_id !== "ollama" && !String(config.api_key || "").trim()) {
    throw new Error(`${config.provider_name} API key is missing for dataset generation.`);
  }

  const chunks = splitCorpus(text, { maxChars: options.maxChars || 1800, overlapChars: options.overlapChars || 180 });
  if (!chunks.length) throw new Error("No usable text chunks were created.");

  const rows = [];
  for (let index = 0; index < chunks.length && rows.length < targetQuestions; index += 1) {
    const remaining = targetQuestions - rows.length;
    const count = Math.min(questionsPerChunk, remaining);
    progress(`[${index + 1}/${chunks.length}] 生成 ${count} 条 QA`);
    const generated = await generateQaForChunk(chunks[index], count, timeout, config);
    for (const item of generated) {
      const query = String(item.query || item.question || "").trim();
      const answerRaw = item.answer ?? item.answers ?? [];
      const answer = (Array.isArray(answerRaw) ? answerRaw : [answerRaw])
        .map((value) => String(value || "").trim())
        .filter(Boolean);
      const quote = String(item.evidence_quote || item.quote || item.evidence || "").trim();
      if (!query || !answer.length || !quote) continue;
      rows.push({
        id: rows.length,
        query,
        answer,
        evidence: [
          {
            source_file: sourceName,
            page: "",
            quote
          }
        ],
        positive: [quote]
      });
      if (rows.length >= targetQuestions) break;
    }
  }
  if (!rows.length) throw new Error("No valid QA items were generated.");

  await ensureDir(DATA_DIR);
  const datasetPath = path.join(DATA_DIR, `${datasetName}.json`);
  const corpusPath = path.join(DATA_DIR, `${datasetName}.corpus.md`);
  const reportPath = path.join(DATA_DIR, `${datasetName}.json.report.json`);

  await fs.writeFile(datasetPath, `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`, "utf8");
  await fs.writeFile(corpusPath, `${text}\n`, "utf8");
  await writeJson(reportPath, {
    dataset_name: datasetName,
    dataset_path: datasetPath,
    corpus_path: corpusPath,
    source_name: sourceName,
    source_chars: text.length,
    chunks: chunks.length,
    qa_count: rows.length,
    generator: {
      runtime: "node",
      provider: config.provider_id,
      model: config.model
    }
  });

  return {
    dataset_name: datasetName,
    dataset_path: datasetPath,
    corpus_path: corpusPath,
    report_path: reportPath,
    qa_count: rows.length,
    rows
  };
}
