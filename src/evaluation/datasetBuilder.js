import fs from "node:fs/promises";
import path from "node:path";
import { ASSETS_DIR, DATA_DIR, MINERU_CACHE_DIR, RUNS_DIR } from "../config.js";
import { chatCompletion, loadLlmConfig, messageText } from "../llm.js";
import { copyFileUnique, ensureDir, pathExists, readJson, readText, sanitizeDatasetName, sanitizeFilename, sha256File, writeJson } from "../utils/files.js";
import { extractJsonObject } from "./judge.js";
import { writeDeepLocalsChunkSidecars } from "./deeplocalsCompat.js";
import { mineruSupportedPaths, processWithMineru } from "./mineru.js";
import { stripThinking } from "./scoring.js";

function compactWhitespace(text) {
  return String(text || "").replace(/\r\n/gu, "\n").replace(/[ \t]+/gu, " ").replace(/\n{3,}/gu, "\n\n").trim();
}

const TEXT_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".markdown",
  ".html",
  ".htm",
  ".csv",
  ".json",
  ".jsonl",
  ".log",
  ".eml",
  ".xml",
  ".yml",
  ".yaml",
  ".py",
  ".js",
  ".ts",
  ".tsx",
  ".css"
]);

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

function datasetAssetDir(datasetName) {
  return path.join(ASSETS_DIR, sanitizeDatasetName(datasetName));
}

function mineruCacheDir(digest) {
  if (!/^[a-f0-9]{64}$/iu.test(digest)) throw new Error(`Invalid sha256 digest: ${digest}`);
  return path.join(MINERU_CACHE_DIR, digest.slice(0, 2), digest);
}

async function copyMineruResultToAssets(result, targetDir) {
  await ensureDir(targetDir);
  const markdownSource = result.markdown_path;
  const markdownTarget = await copyFileUnique(markdownSource, targetDir, path.basename(markdownSource));
  for (const [source, suffix] of [
    [result.metadata_path, ".meta.json"],
    [result.combined_json_path, ".mineru.json"]
  ]) {
    if (source && (await pathExists(source))) {
      await fs.copyFile(source, `${markdownTarget}${suffix}`);
    }
  }
  for (const source of result.zip_paths || []) {
    if (await pathExists(source)) await copyFileUnique(source, path.join(targetDir, "raw_zips"), path.basename(source));
  }
  for (const source of result.json_paths || []) {
    if (await pathExists(source)) await copyFileUnique(source, path.join(targetDir, "raw_json"), path.basename(source));
  }
  return markdownTarget;
}

async function storeMineruResultInCache(result, digest, originalPath) {
  const cacheDir = mineruCacheDir(digest);
  await fs.rm(cacheDir, { recursive: true, force: true });
  await ensureDir(cacheDir);
  const originalCopy = await copyFileUnique(originalPath, path.join(cacheDir, "original"), path.basename(originalPath));
  const markdownPath = await copyMineruResultToAssets(result, path.join(cacheDir, "mineru"));
  const manifest = {
    schema: "rageval-mineru-cache-node-v1",
    file_sha256: digest,
    original_name: path.basename(originalPath),
    created_at: new Date().toISOString(),
    original_files: [originalCopy],
    markdown_path: markdownPath,
    metadata_path: `${markdownPath}.meta.json`,
    semantic_json_path: `${markdownPath}.mineru.json`,
    raw_json_files: await listFiles(path.join(cacheDir, "mineru", "raw_json"), ".json"),
    raw_zip_files: await listFiles(path.join(cacheDir, "mineru", "raw_zips"), ".zip")
  };
  await writeJson(path.join(cacheDir, "manifest.json"), manifest);
  return markdownPath;
}

async function listFiles(dirPath, suffix = "") {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile() && (!suffix || entry.name.toLowerCase().endsWith(suffix)))
      .map((entry) => path.join(dirPath, entry.name))
      .sort();
  } catch {
    return [];
  }
}

async function loadMineruCacheEntry(digest) {
  const manifestPath = path.join(mineruCacheDir(digest), "manifest.json");
  const entry = await readJson(manifestPath, null);
  if (!entry?.markdown_path || !(await pathExists(entry.markdown_path))) return null;
  return entry;
}

async function copyCachedMineruToAssets(cacheEntry, targetDir) {
  await ensureDir(targetDir);
  const markdownTarget = await copyFileUnique(cacheEntry.markdown_path, targetDir, path.basename(cacheEntry.markdown_path));
  for (const [source, suffix] of [
    [cacheEntry.metadata_path, ".meta.json"],
    [cacheEntry.semantic_json_path, ".mineru.json"]
  ]) {
    if (source && (await pathExists(source))) await fs.copyFile(source, `${markdownTarget}${suffix}`);
  }
  for (const source of cacheEntry.raw_json_files || []) {
    if (await pathExists(source)) await copyFileUnique(source, path.join(targetDir, "raw_json"), path.basename(source));
  }
  for (const source of cacheEntry.raw_zip_files || []) {
    if (await pathExists(source)) await copyFileUnique(source, path.join(targetDir, "raw_zips"), path.basename(source));
  }
  return markdownTarget;
}

async function readTextInput(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (!TEXT_EXTENSIONS.has(ext)) {
    throw new Error(`${path.basename(filePath)} is not a text input. Enable MinerU for PDF/Office/image parsing.`);
  }
  let text = await readText(filePath);
  if ([".html", ".htm"].includes(ext)) {
    text = text
      .replace(/<style[\s\S]*?<\/style>/giu, " ")
      .replace(/<script[\s\S]*?<\/script>/giu, " ")
      .replace(/<[^>]+>/gu, " ")
      .replace(/&nbsp;/gu, " ")
      .replace(/&amp;/gu, "&")
      .replace(/&lt;/gu, "<")
      .replace(/&gt;/gu, ">")
      .replace(/&quot;/gu, '"')
      .replace(/&#39;/gu, "'");
  }
  return compactWhitespace(text);
}

async function writeAssetManifest(assetDir, payload) {
  const manifest = {
    schema: "rageval-dataset-assets-node-v1",
    created_at: new Date().toISOString(),
    ...payload
  };
  const manifestPath = path.join(assetDir, "manifest.json");
  await writeJson(manifestPath, manifest);
  return manifestPath;
}

export async function buildDatasetFromFiles(options, progress = () => {}) {
  const datasetName = sanitizeDatasetName(options.datasetName);
  const files = (options.files || [])
    .map((file) => ({
      path: file.path,
      originalName: sanitizeFilename(file.originalName || path.basename(file.path))
    }))
    .filter((file) => file.path);
  if (!files.length) throw new Error("No uploaded files were provided.");

  const runDir = options.runDir || path.join(RUNS_DIR, `${datasetName}_${Date.now()}`);
  const assetDir = datasetAssetDir(datasetName);
  const originalDir = path.join(assetDir, "original");
  const mineruDir = path.join(assetDir, "mineru");
  await ensureDir(runDir);
  await ensureDir(originalDir);
  await ensureDir(mineruDir);
  await ensureDir(MINERU_CACHE_DIR);

  progress(`保存原始文件：${originalDir}`);
  const inputPaths = [];
  for (const file of files) {
    inputPaths.push(await copyFileUnique(file.path, originalDir, file.originalName));
  }

  const fileHashes = new Map();
  for (const filePath of inputPaths) {
    fileHashes.set(filePath, await sha256File(filePath));
  }

  const useMineru = options.useMineru !== false;
  const mineruCandidates = mineruSupportedPaths(inputPaths);
  const localCandidates = inputPaths.filter((filePath) => !mineruCandidates.includes(filePath));
  const builderInputs = [];
  const mineruReport = [];

  if (useMineru && mineruCandidates.length) {
    const uncached = [];
    for (const candidate of mineruCandidates) {
      const digest = fileHashes.get(candidate);
      const cacheEntry = await loadMineruCacheEntry(digest);
      if (cacheEntry) {
        const assetMarkdown = await copyCachedMineruToAssets(cacheEntry, mineruDir);
        builderInputs.push(assetMarkdown);
        mineruReport.push({
          original_path: candidate,
          success: true,
          cache_hit: true,
          file_sha256: digest,
          markdown_path: assetMarkdown,
          metadata_path: `${assetMarkdown}.meta.json`,
          combined_json_path: `${assetMarkdown}.mineru.json`
        });
        progress(`复用 MinerU 缓存：${path.basename(candidate)} -> ${digest.slice(0, 12)}`);
      } else {
        uncached.push(candidate);
      }
    }

    progress(`MinerU 解析文件数：${uncached.length}，缓存命中：${mineruCandidates.length - uncached.length}`);
    if (uncached.length) {
      const mineruOutputDir = path.join(runDir, "mineru_markdown");
      const results = await processWithMineru(
        uncached,
        mineruOutputDir,
        {
          token: options.mineruToken || "",
          apiBase: options.mineruApiBase || "",
          modelVersion: options.mineruModel || "vlm",
          language: options.language || "ch",
          isOcr: options.isOcr !== false,
          enableFormula: options.enableFormula !== false,
          enableTable: options.enableTable !== false,
          timeout: options.timeout || 300
        },
        progress
      );
      for (const result of results) {
        mineruReport.push(result);
        if (result.success && result.markdown_path) {
          const originalPath = inputPaths.find((item) => path.basename(item) === path.basename(result.original_path)) || result.original_path;
          const digest = fileHashes.get(originalPath) || (await sha256File(originalPath));
          await storeMineruResultInCache(result, digest, originalPath);
          const assetMarkdown = await copyMineruResultToAssets(result, mineruDir);
          builderInputs.push(assetMarkdown);
          progress(`写入 MinerU 缓存：${path.basename(originalPath)} -> ${digest.slice(0, 12)}`);
        } else {
          progress(`MinerU 失败：${path.basename(result.original_path)} - ${result.error}`);
        }
      }
    }
  } else {
    if (mineruCandidates.length && !useMineru) progress("已关闭 MinerU，尝试使用本地文本解析");
    localCandidates.push(...mineruCandidates);
  }

  for (const localPath of localCandidates) {
    const text = await readTextInput(localPath);
    if (text) {
      const target = await copyFileUnique(localPath, path.join(assetDir, "text"), path.basename(localPath));
      builderInputs.push(target);
    }
  }

  if (!builderInputs.length) throw new Error("没有可用于生成数据集的解析结果");

  const mineruMarkdownInputs = builderInputs.filter((item) => [".md", ".markdown"].includes(path.extname(item).toLowerCase()));
  const deepLocalsChunkSidecars = await writeDeepLocalsChunkSidecars(mineruMarkdownInputs, datasetName, progress);
  if (mineruMarkdownInputs.length && !deepLocalsChunkSidecars.length) {
    progress("未生成 DeepLocals 分块 sidecar，将在真实测评时使用普通 corpus 上传");
  }

  const corpusSections = [];
  for (const inputPath of builderInputs) {
    const text = await readText(inputPath);
    if (text.trim()) corpusSections.push(`<!-- source: ${path.basename(inputPath)} -->\n\n${text.trim()}`);
  }
  const corpusText = compactWhitespace(corpusSections.join("\n\n"));
  if (!corpusText) throw new Error("解析结果为空，无法生成数据集");

  progress(`解析完成：${builderInputs.length} 个输入，开始生成 QA`);
  const result = await buildDatasetFromText(
    {
      datasetName,
      sourceName: `${datasetName}.corpus.md`,
      text: corpusText,
      targetQuestions: options.targetQuestions,
      questionsPerChunk: options.questionsPerChunk,
      timeout: options.timeout,
      llmConfig: options.llmConfig
    },
    progress
  );

  const manifestPath = await writeAssetManifest(assetDir, {
    dataset_name: datasetName,
    original_files: inputPaths,
    mineru_markdown_files: mineruMarkdownInputs,
    deepseekmine_chunk_files: deepLocalsChunkSidecars,
    deeplocals_chunk_files: deepLocalsChunkSidecars,
    raw_mineru_json_files: await listFiles(path.join(mineruDir, "raw_json"), ".json"),
    raw_mineru_zip_files: await listFiles(path.join(mineruDir, "raw_zips"), ".zip"),
    dataset_path: result.dataset_path,
    corpus_path: result.corpus_path,
    mineru_report: mineruReport,
    config: {
      use_mineru: useMineru,
      mineru_model: options.mineruModel || "vlm",
      is_ocr: options.isOcr !== false,
      enable_table: options.enableTable !== false,
      enable_formula: options.enableFormula !== false,
      target_questions: options.targetQuestions,
      questions_per_chunk: options.questionsPerChunk
    }
  });

  const report = await readJson(result.report_path, {});
  await writeJson(result.report_path, {
    ...report,
    assets_dir: assetDir,
    assets_manifest: manifestPath,
    builder_inputs: builderInputs,
    deepseekmine_chunk_sidecars: deepLocalsChunkSidecars,
    deeplocals_chunk_sidecars: deepLocalsChunkSidecars,
    mineru: mineruReport
  });

  return {
    ...result,
    assets_dir: assetDir,
    assets_manifest: manifestPath,
    builder_inputs: builderInputs,
    deepseekmine_chunk_sidecars: deepLocalsChunkSidecars,
    deeplocals_chunk_sidecars: deepLocalsChunkSidecars,
    mineru: mineruReport
  };
}
