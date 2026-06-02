import http from "node:http";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import Busboy from "busboy";
import { ASSETS_DIR, DATA_DIR, RESULT_ZH_DIR, RUNS_DIR } from "./config.js";
import { runPlatformCompare } from "./compare.js";
import { runCherryEval } from "./evaluation/cherry.js";
import { buildDatasetFromFiles, buildDatasetFromText } from "./evaluation/datasetBuilder.js";
import { runDeepLocalsEval } from "./evaluation/deeplocals.js";
import {
  configFromPayload,
  listProviderModels,
  loadLlmConfig,
  loadOllamaContextConfig,
  maskedConfig,
  preserveSavedApiKey,
  providerPresetsPayload,
  saveLlmConfig,
  saveOllamaContextConfig
} from "./llm.js";
import { ensureDir, pathExists, readJson, readText, sanitizeDatasetName, sanitizeFilename } from "./utils/files.js";

const jobs = new Map();

class JobCancelledError extends Error {
  constructor(message = "任务已取消") {
    super(message);
    this.name = "JobCancelledError";
  }
}

function jsonResponse(res, payload, status = 200) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

function textResponse(res, body, contentType = "text/html; charset=utf-8", status = 200) {
  res.writeHead(status, {
    "Content-Type": contentType,
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

async function fileResponse(res, filePath) {
  const content = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = {
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jsonl": "application/jsonl; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".zip": "application/zip"
  }[ext] || "application/octet-stream";
  res.writeHead(200, {
    "Content-Type": contentType,
    "Content-Length": content.length,
    "Content-Disposition": `attachment; filename="${path.basename(filePath).replace(/"/gu, "")}"`
  });
  res.end(content);
}

async function readBodyJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
}

function fieldBool(value, fallback = false) {
  if (value === undefined || value === null || value === "") return fallback;
  return ["1", "true", "on", "yes"].includes(String(value).toLowerCase());
}

async function readMultipartForm(req) {
  await ensureDir(RUNS_DIR);
  const runDir = path.join(RUNS_DIR, `upload_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`);
  const uploadDir = path.join(runDir, "uploads");
  await ensureDir(uploadDir);
  return new Promise((resolve, reject) => {
    const fields = {};
    const files = [];
    const writes = [];
    const busboy = Busboy({ headers: req.headers });
    busboy.on("field", (name, value) => {
      fields[name] = value;
    });
    busboy.on("file", (name, stream, info) => {
      const filename = sanitizeFilename(info?.filename || "");
      if (!filename) {
        stream.resume();
        return;
      }
      const target = path.join(uploadDir, `${String(files.length + 1).padStart(3, "0")}_${filename}`);
      files.push({ fieldName: name, path: target, originalName: filename, mimeType: info?.mimeType || "" });
      writes.push(pipeline(stream, createWriteStream(target)));
    });
    busboy.on("error", reject);
    busboy.on("close", () => {
      Promise.all(writes)
        .then(() => resolve({ fields, files, runDir }))
        .catch(reject);
    });
    req.pipe(busboy);
  });
}

function normalizeBuildFileOptions(fields, files, runDir) {
  return {
    datasetName: fields.dataset_name || fields.datasetName || "custom_rag_eval",
    files,
    runDir,
    useMineru: fieldBool(fields.use_mineru ?? fields.useMineru, true),
    mineruModel: fields.mineru_model || fields.mineruModel || "vlm",
    mineruToken: fields.mineru_token || fields.mineruToken || "",
    language: fields.language || "ch",
    isOcr: fieldBool(fields.is_ocr ?? fields.isOcr, true),
    enableTable: fieldBool(fields.enable_table ?? fields.enableTable, true),
    enableFormula: fieldBool(fields.enable_formula ?? fields.enableFormula, true),
    targetQuestions: Number.parseInt(String(fields.target_count || fields.targetQuestions || 20), 10) || 20,
    questionsPerChunk: Number.parseInt(String(fields.questions_per_chunk || fields.questionsPerChunk || 3), 10) || 3,
    timeout: Number.parseInt(String(fields.timeout || 300), 10) || 300
  };
}

async function countJsonl(filePath) {
  const text = await readText(filePath);
  return text.split(/\r?\n/gu).filter((line) => line.trim()).length;
}

async function listDatasets() {
  await ensureDir(DATA_DIR);
  const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
  const datasets = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json") || entry.name.endsWith(".report.json")) continue;
    const filePath = path.join(DATA_DIR, entry.name);
    const name = path.basename(entry.name, ".json");
    const stat = await fs.stat(filePath);
    const report = await readJson(path.join(DATA_DIR, `${name}.json.report.json`), null);
    const corpusCandidates = [
      path.join(DATA_DIR, `${name}.corpus.md`),
      path.join(DATA_DIR, `${name}.corpus.txt`),
      ...(report && typeof report === "object"
        ? ["corpus_path", "corpus", "corpus_source"].map((key) => String(report[key] || "")).filter(Boolean)
        : [])
    ];
    let corpusPath = "";
    for (const candidate of corpusCandidates) {
      const resolved = path.resolve(candidate);
      if (await pathExists(resolved)) {
        corpusPath = resolved;
        break;
      }
    }
    const assetsDir = path.join(DATA_DIR, "assets", name);
    const hasAssets = await pathExists(assetsDir);
    datasets.push({
      name,
      path: filePath,
      rows: await countJsonl(filePath).catch(() => 0),
      has_corpus: Boolean(corpusPath),
      corpus_path: corpusPath,
      has_assets: hasAssets,
      updated_at: stat.mtime.toISOString()
    });
  }
  return datasets.sort((a, b) => a.name.localeCompare(b.name));
}

async function listReports() {
  await ensureDir(RESULT_ZH_DIR);
  const entries = await fs.readdir(RESULT_ZH_DIR, { withFileTypes: true }).catch(() => []);
  const reports = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".json")) continue;
    const filePath = path.join(RESULT_ZH_DIR, entry.name);
    const stat = await fs.stat(filePath);
    const payload = await readJson(filePath, {});
    const summary = payload?.summary || {};
    const displayName = entry.name.replace(/^deepseekmine_/iu, "deeplocals_");
    reports.push({
      name: entry.name,
      display_name: displayName,
      path: filePath,
      dataset: payload?.dataset_name || payload?.dataset || "",
      schema: payload?.schema || "",
      qa_accuracy: summary.qa_accuracy ?? null,
      qa_correct: summary.qa_correct ?? null,
      qa_total: summary.qa_total ?? summary.total ?? null,
      updated_at: stat.mtime.toISOString()
    });
  }
  return reports.sort((a, b) => b.updated_at.localeCompare(a.updated_at)).slice(0, 80);
}

function assertDatasetName(rawName) {
  const value = String(rawName || "").trim();
  const cleaned = sanitizeDatasetName(value, "");
  if (!value || cleaned !== value || value === "." || value === "..") {
    throw new Error("Invalid dataset name.");
  }
  return cleaned;
}

function assertInside(filePath, rootDir) {
  const resolvedRoot = path.resolve(rootDir);
  const resolvedPath = path.resolve(filePath);
  const relative = path.relative(resolvedRoot, resolvedPath);
  if (relative && (relative.startsWith("..") || path.isAbsolute(relative))) {
    throw new Error(`Unsafe path: ${resolvedPath}`);
  }
  return resolvedPath;
}

function resolveDownloadPath(filePath) {
  const roots = [DATA_DIR, RESULT_ZH_DIR, RUNS_DIR].map((root) => path.resolve(root));
  const resolved = path.resolve(String(filePath || ""));
  for (const root of roots) {
    const relative = path.relative(root, resolved);
    if (!relative || (!relative.startsWith("..") && !path.isAbsolute(relative))) return resolved;
  }
  throw new Error(`Download not allowed: ${resolved}`);
}

async function deletePathIfExists(filePath, rootDir, deleted) {
  const resolved = assertInside(filePath, rootDir);
  if (!(await pathExists(resolved))) return;
  await fs.rm(resolved, { recursive: true, force: true });
  deleted.push(resolved);
}

function reportFilenameMatchesDataset(filename, datasetName) {
  return filename.includes(`_${datasetName}_`) || filename.startsWith(`${datasetName}_`);
}

function downloadUrl(filePath) {
  return filePath ? `/download?path=${encodeURIComponent(filePath)}` : "";
}

function decorateJobResult(job, result) {
  const outputPath = result?.output_path || result?.dataset_path || "";
  const reportPath = result?.report_path || result?.summary_path || "";
  job.output_path = outputPath;
  job.report_path = reportPath;
  job.summary_path = result?.summary_path || "";
  job.csv_path = result?.csv_path || "";
  job.corpus_path = result?.corpus_path || result?.selected_corpus_path || "";
  job.assets_manifest = result?.assets_manifest || "";
  job.output_url = downloadUrl(outputPath);
  job.report_url = downloadUrl(reportPath);
  job.summary_url = downloadUrl(job.summary_path);
  job.csv_url = downloadUrl(job.csv_path);
  job.corpus_url = downloadUrl(job.corpus_path);
}

async function deleteDataset(datasetName) {
  const name = assertDatasetName(datasetName);
  const deleted = [];
  for (const filePath of [
    path.join(DATA_DIR, `${name}.json`),
    path.join(DATA_DIR, `${name}.corpus.txt`),
    path.join(DATA_DIR, `${name}.corpus.md`),
    path.join(DATA_DIR, `${name}.corpus.generated.md`),
    path.join(DATA_DIR, `${name}.json.report.json`)
  ]) {
    await deletePathIfExists(filePath, DATA_DIR, deleted);
  }
  await deletePathIfExists(path.join(ASSETS_DIR, name), ASSETS_DIR, deleted);

  const entries = await fs.readdir(RESULT_ZH_DIR, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (!entry.isFile() || !/\.(json|csv)$/iu.test(entry.name)) continue;
    const reportPath = assertInside(path.join(RESULT_ZH_DIR, entry.name), RESULT_ZH_DIR);
    let shouldDelete = reportFilenameMatchesDataset(entry.name, name);
    if (!shouldDelete && entry.name.endsWith(".json")) {
      const payload = await readJson(reportPath, null);
      shouldDelete = payload?.dataset_name === name || payload?.dataset === name;
    }
    if (shouldDelete) {
      await fs.rm(reportPath, { force: true });
      deleted.push(reportPath);
    }
  }

  if (!deleted.length) throw new Error(`未找到数据集：${name}`);
  return { ok: true, dataset: name, deleted };
}

function createJob(kind, runner) {
  const id = `${kind}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  const job = {
    id,
    kind,
    status: "running",
    progress: 5,
    logs: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: null,
    error: "",
    output_path: "",
    report_path: "",
    output_url: "",
    report_url: "",
    summary_path: "",
    summary_url: "",
    csv_path: "",
    csv_url: "",
    corpus_path: "",
    corpus_url: "",
    assets_manifest: "",
    cancel_requested: false
  };
  jobs.set(id, job);
  const step = { build_dataset: 4, deeplocals: 3, cherry: 4, compare: 2 }[kind] || 3;
  const checkCancelled = () => {
    if (job.cancel_requested) throw new JobCancelledError();
  };
  const log = (message) => {
    checkCancelled();
    job.logs.push({ at: new Date().toISOString(), message: String(message) });
    if (job.logs.length > 400) job.logs.shift();
    if (job.status === "running") job.progress = Math.min(95, Math.max(job.progress, 5 + job.logs.length * step));
    job.updated_at = new Date().toISOString();
  };
  log.setProgress = (value) => {
    checkCancelled();
    const next = Number.parseInt(String(value), 10);
    if (Number.isInteger(next)) job.progress = Math.max(0, Math.min(99, next));
    job.updated_at = new Date().toISOString();
  };
  log.checkCancelled = checkCancelled;
  Promise.resolve()
    .then(() => runner(log))
    .then((result) => {
      job.status = job.cancel_requested ? "cancelled" : "completed";
      job.progress = 100;
      job.result = result;
      if (result && typeof result === "object") decorateJobResult(job, result);
      job.updated_at = new Date().toISOString();
    })
    .catch((error) => {
      job.status = error instanceof JobCancelledError ? "cancelled" : "failed";
      job.progress = 100;
      job.error = error.stack || error.message || String(error);
      job.updated_at = new Date().toISOString();
    });
  return job;
}

function cancelJob(id) {
  const job = jobs.get(id);
  if (!job) return { error: "job not found" };
  if (["completed", "failed", "cancelled"].includes(job.status)) return { ok: true, status: job.status };
  job.cancel_requested = true;
  job.status = "cancelling";
  job.logs.push({ at: new Date().toISOString(), message: "已请求取消，等待当前步骤安全停下" });
  job.updated_at = new Date().toISOString();
  return { ok: true, status: job.status };
}

function normalizeEvalOptions(payload) {
  return {
    datasetName: payload.dataset || payload.datasetName || "zh_int_clean",
    datasetPath: payload.datasetPath || "",
    corpusPath: payload.corpusPath || "",
    apiBase: payload.apiBase || payload.deeplocalApiBase || "http://127.0.0.1:3335",
    knowledgeLabel: payload.knowledgeLabel || "",
    mode: payload.mode || "qa",
    limit: Number.parseInt(String(payload.limit || 0), 10) || 0,
    ids: payload.ids || "",
    output: payload.output || "",
    includePrompts: !payload.noPrompts,
    useAdaptiveRag: !payload.noAdaptiveRag,
    timeout: Number.parseInt(String(payload.timeout || 900), 10) || 900
  };
}

function normalizeCherryOptions(payload) {
  return {
    datasetName: payload.dataset || payload.datasetName || "zh_int_clean",
    datasetPath: payload.datasetPath || "",
    apiBase: payload.apiBase || payload.cherryApiBase || "http://127.0.0.1:23333",
    apiKey: payload.apiKey || payload.cherryApiKey || "",
    knowledgeBaseIds: payload.knowledgeBaseId || payload.knowledgeBaseIds || payload.cherryKnowledgeBaseId || "",
    model: payload.model || payload.cherryModel || "",
    documentCount: Number.parseInt(String(payload.documentCount || 20), 10) || 20,
    limit: Number.parseInt(String(payload.limit || 0), 10) || 0,
    ids: payload.ids || "",
    output: payload.output || "",
    timeout: Number.parseInt(String(payload.timeout || 300), 10) || 300,
    judgeMode: payload.judgeMode || "rule_then_model",
    temperature: Number.parseFloat(String(payload.temperature || 0)) || 0,
    maxTokens: Number.parseInt(String(payload.maxTokens || 1024), 10) || 1024
  };
}

function normalizeCompareOptions(payload) {
  return {
    dataset: payload.dataset || "zh_int_clean",
    limit: Number.parseInt(String(payload.limit || 0), 10) || 0,
    ids: payload.ids || "",
    timeout: Number.parseInt(String(payload.timeout || 900), 10) || 900,
    documentCount: Number.parseInt(String(payload.documentCount || 20), 10) || 20,
    deeplocalApiBase: payload.deeplocalApiBase || "http://127.0.0.1:3335",
    cherryApiBase: payload.cherryApiBase || "http://127.0.0.1:23333",
    cherryApiKey: payload.cherryApiKey || payload.apiKey || "",
    cherryKnowledgeBaseId: payload.cherryKnowledgeBaseId || payload.knowledgeBaseId || "",
    cherryModel: payload.cherryModel || "silicon:deepseek-ai/DeepSeek-V4-Flash",
    deeplocalAnswerProvider: payload.deeplocalAnswerProvider || "siliconflow",
    deeplocalAnswerModel: payload.deeplocalAnswerModel || "deepseek-ai/DeepSeek-V4-Flash",
    deeplocalAnswerApiUrl: payload.deeplocalAnswerApiUrl || "",
    deeplocalAnswerApiKey: payload.deeplocalAnswerApiKey || "",
    outputPrefix: payload.outputPrefix || "",
    noPrompts: Boolean(payload.noPrompts)
  };
}

async function route(req, res) {
  const url = new URL(req.url || "/", "http://localhost");
  if (req.method === "GET" && url.pathname === "/") return textResponse(res, indexHtml());
  if (req.method === "GET" && url.pathname === "/download") {
    const target = resolveDownloadPath(url.searchParams.get("path") || "");
    if (!(await pathExists(target))) return jsonResponse(res, { error: "file not found" }, 404);
    return fileResponse(res, target);
  }
  if (req.method === "GET" && url.pathname === "/api/datasets") return jsonResponse(res, { datasets: await listDatasets() });
  if (req.method === "GET" && url.pathname === "/api/reports") return jsonResponse(res, { reports: await listReports() });
  if (req.method === "GET" && url.pathname === "/api/llm-config") {
    const config = await loadLlmConfig();
    return jsonResponse(res, { config: maskedConfig(config), presets: providerPresetsPayload() });
  }
  if (req.method === "GET" && url.pathname === "/api/ollama/context-config") {
    return jsonResponse(res, await loadOllamaContextConfig());
  }
  if (req.method === "GET" && url.pathname.startsWith("/api/jobs/")) {
    const id = decodeURIComponent(url.pathname.split("/").pop() || "");
    const job = jobs.get(id);
    return jsonResponse(res, job ? { job } : { error: "job not found" }, job ? 200 : 404);
  }
  if (req.method === "POST" && url.pathname.startsWith("/api/jobs/") && url.pathname.endsWith("/cancel")) {
    const id = decodeURIComponent(url.pathname.split("/")[3] || "");
    const result = cancelJob(id);
    return jsonResponse(res, result, result.error ? 404 : 200);
  }
  if (req.method === "POST" && url.pathname === "/api/ollama/context-config") {
    const payload = await readBodyJson(req);
    return jsonResponse(res, { success: true, ...(await saveOllamaContextConfig(payload.context_length)) });
  }
  if (req.method === "POST" && url.pathname === "/api/llm-config") {
    const payload = await readBodyJson(req);
    const config = await preserveSavedApiKey(await configFromPayload(payload));
    await saveLlmConfig(config);
    return jsonResponse(res, { success: true, config: maskedConfig(config), presets: providerPresetsPayload() });
  }
  if (req.method === "POST" && url.pathname === "/api/llm-config/test") {
    const payload = await readBodyJson(req);
    const config = await preserveSavedApiKey(await configFromPayload(payload));
    const models = await listProviderModels(config, 20);
    const tested = { ...config, models, model: models.includes(config.model) ? config.model : models[0] };
    return jsonResponse(res, { success: true, models, config: maskedConfig(tested), message: `发现 ${models.length} 个可用模型` });
  }
  if (req.method === "DELETE" && url.pathname.startsWith("/api/datasets/")) {
    const name = decodeURIComponent(url.pathname.split("/").pop() || "");
    return jsonResponse(res, await deleteDataset(name));
  }
  if (req.method === "POST" && url.pathname === "/api/datasets/build") {
    const contentType = String(req.headers["content-type"] || "");
    if (contentType.includes("multipart/form-data")) {
      const { fields, files, runDir } = await readMultipartForm(req);
      const options = normalizeBuildFileOptions(fields, files, runDir);
      const job = createJob("build_dataset", (log) => buildDatasetFromFiles(options, log));
      return jsonResponse(res, { job_id: job.id, job });
    }
    const payload = await readBodyJson(req);
    const job = createJob("build_dataset", (log) => buildDatasetFromText(payload, log));
    return jsonResponse(res, { job_id: job.id, job });
  }
  if (req.method === "POST" && url.pathname === "/api/eval/deeplocals") {
    const payload = await readBodyJson(req);
    const job = createJob("deeplocals", (log) => runDeepLocalsEval(normalizeEvalOptions(payload), log));
    return jsonResponse(res, { job_id: job.id, job });
  }
  if (req.method === "POST" && url.pathname === "/api/eval/cherry") {
    const payload = await readBodyJson(req);
    const job = createJob("cherry", (log) => runCherryEval(normalizeCherryOptions(payload), log));
    return jsonResponse(res, { job_id: job.id, job });
  }
  if (req.method === "POST" && url.pathname === "/api/eval/compare") {
    const payload = await readBodyJson(req);
    const job = createJob("compare", (log) => runPlatformCompare(normalizeCompareOptions(payload), log));
    return jsonResponse(res, { job_id: job.id, job });
  }
  return jsonResponse(res, { error: "not found" }, 404);
}

export function startServer({ host = "127.0.0.1", port = 7861 } = {}) {
  const server = http.createServer((req, res) => {
    route(req, res).catch((error) => jsonResponse(res, { error: error.stack || error.message }, 500));
  });
  server.listen(port, host, () => {
    console.log(`RAGEval Forge Node UI: http://${host}:${port}/`);
  });
  return server;
}

function indexHtml() {
  return String.raw`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAGEval Forge</title>
  <style>
    :root {
      color-scheme: light;
      --bg:#f4f6f8; --surface:#fff; --surface-2:#f8fafc; --text:#111827; --muted:#64748b;
      --line:#d8e0ea; --line-strong:#c6d0dd; --accent:#0f766e; --accent-strong:#0b5f59;
      --blue:#2563eb; --bad:#b42318; --radius:8px;
    }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--text); background:var(--bg); }
    .app-header {
      min-height:88px; display:flex; align-items:center; justify-content:space-between; gap:24px;
      padding:20px 32px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:3;
    }
    .brand { display:flex; align-items:center; gap:14px; }
    .brand-mark {
      width:38px; height:38px; border-radius:8px; display:grid; place-items:center;
      background:#102a43; color:#fff; font-weight:800; box-shadow:inset 0 -1px 0 rgba(255,255,255,.16);
    }
    h1 { font-size:20px; margin:0 0 2px; line-height:1.2; }
    .brand p { margin:0; color:var(--muted); font-size:13px; }
    .top-pills { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .pill {
      display:inline-flex; align-items:center; gap:7px; height:32px; padding:0 12px; border:1px solid var(--line);
      border-radius:7px; background:#fbfdff; color:#334155; font-size:12px; font-weight:650;
    }
    .dot { width:7px; height:7px; border-radius:999px; background:var(--accent); }
    .shell {
      display:grid; grid-template-columns:minmax(420px,560px) minmax(460px,1fr); gap:22px;
      max-width:1480px; margin:0 auto; padding:24px;
    }
    .panel { background:var(--surface); border:1px solid var(--line); border-radius:var(--radius); overflow:hidden; }
    .workspace-tabs {
      display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; padding:12px; border-bottom:1px solid var(--line); background:#f8fafc;
    }
    .workspace-tab {
      height:40px; border:1px solid var(--line); border-radius:6px; background:#fff; color:#334155; font-weight:750; cursor:pointer;
    }
    .workspace-tab.active { background:var(--accent); color:#fff; border-color:var(--accent); }
    .workflow-panel { display:none; }
    .workflow-panel.active { display:block; }
    .section-head { padding:22px 22px 18px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#fff,#fbfcfe); }
    .section-head h2, .monitor-head h2 { margin:4px 0 6px; font-size:18px; }
    .section-head p, .monitor-head p { margin:0; color:var(--muted); line-height:1.55; }
    .kicker { color:#08736b; font-size:12px; text-transform:uppercase; font-weight:850; letter-spacing:0; }
    .panel-pad, .run-body { padding:22px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .triple-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    label { display:block; font-size:12px; font-weight:760; color:#334155; margin-bottom:6px; }
    input, textarea, select {
      width:100%; min-height:40px; border:1px solid #bdcad8; border-radius:6px; padding:9px 10px;
      font:inherit; background:#fff; color:var(--text);
    }
    textarea { min-height:180px; resize:vertical; }
    .field { margin-bottom:14px; }
    .file-drop {
      display:block; border:1px dashed #aab6c5; border-radius:var(--radius); background:#f8fafc;
      padding:18px; cursor:pointer; transition:border-color .16s ease, background .16s ease;
    }
    .file-drop:hover { border-color:var(--accent); background:#f5fbfa; }
    .file-drop input { width:100%; height:auto; border:0; padding:0; margin-top:12px; background:transparent; }
    .file-title { display:block; font-size:14px; font-weight:760; color:#1f2937; }
    .file-subtitle { display:block; margin-top:4px; font-size:12px; color:var(--muted); line-height:1.5; }
    .toggle-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:4px 0 18px; }
    .checkrow {
      display:flex; align-items:center; gap:10px; min-height:38px; padding:9px 10px;
      border:1px solid var(--line); border-radius:6px; background:#fbfdff; font-size:13px; color:#334155; margin:0;
    }
    .checkrow input { width:16px; min-height:16px; height:16px; margin:0; }
    .dataset-picker { position:relative; margin-bottom:14px; }
    .picker-button {
      width:100%; min-height:66px; height:auto; padding:12px 14px; display:grid; grid-template-columns:1fr auto;
      align-items:center; text-align:left;
    }
    .picker-title {
      display:block; font-size:15px; font-weight:760; line-height:1.35; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }
    .picker-meta { display:block; margin-top:4px; color:var(--muted); font-size:12px; line-height:1.45; }
    .chevron { font-weight:800; color:#64748b; padding-left:12px; }
    .picker-menu {
      display:none; position:absolute; inset:auto 0 auto 0; z-index:6; margin-top:6px; padding:10px;
      border:1px solid var(--line-strong); border-radius:8px; background:#fff; box-shadow:0 14px 32px rgba(15,23,42,.16);
    }
    .picker-menu.open { display:block; }
    .picker-search { height:36px; min-height:36px; margin-bottom:9px; }
    .dataset-list { max-height:278px; overflow:auto; display:grid; gap:7px; }
    .dataset-option {
      width:100%; min-height:58px; padding:10px 11px; border:1px solid var(--line); border-radius:6px;
      background:#fff; display:grid; grid-template-columns:minmax(0,1fr) auto; gap:10px; align-items:center;
    }
    .dataset-option:hover { border-color:#8fb5d8; background:#f8fbff; }
    .dataset-option.selected { border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent); }
    .dataset-option.unavailable { opacity:.58; cursor:not-allowed; background:#f8fafc; }
    .dataset-main {
      min-width:0; border:0; background:transparent; padding:0; text-align:left; cursor:pointer; color:inherit; font:inherit;
    }
    .dataset-main:disabled { cursor:not-allowed; }
    .dataset-name { display:block; font-weight:800; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .dataset-meta { display:block; margin-top:3px; color:var(--muted); font-size:12px; }
    .dataset-actions { display:flex; align-items:center; gap:8px; }
    .tag { border-radius:999px; padding:4px 8px; font-size:12px; font-weight:800; background:#dcfce7; color:#166534; white-space:nowrap; }
    .tag.warn { background:#fef3c7; color:#92400e; }
    .icon-danger {
      width:30px; height:30px; min-height:30px; padding:0; border:1px solid #fecaca; border-radius:6px;
      background:#fff5f5; color:#b42318; font-size:18px; line-height:1; font-weight:850; cursor:pointer;
    }
    .icon-danger:hover { background:#fee2e2; border-color:#fca5a5; }
    .empty-state { padding:18px 10px; color:var(--muted); text-align:center; }
    .metric-strip { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin:14px 0 16px; }
    .metric { border:1px solid var(--line); border-radius:6px; padding:10px; background:#fbfdff; }
    .metric span { display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }
    .metric strong { display:block; font-size:16px; }
    .compare-hero {
      display:grid; grid-template-columns:minmax(0,1fr) auto; gap:14px; align-items:center;
      padding:16px; border:1px solid var(--line); border-radius:var(--radius); background:#fbfdff; margin-bottom:14px;
    }
    .compare-cards, .secret-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .compare-card { border:1px solid var(--line); border-radius:6px; padding:12px; background:#fff; }
    .compare-card span { display:block; color:var(--muted); font-size:11px; margin-bottom:4px; }
    .compare-card strong { display:block; font-size:14px; }
    button.primary {
      background:var(--accent); color:#fff; border:0; border-radius:6px; padding:10px 14px; min-height:42px; font-weight:800; cursor:pointer;
    }
    button.secondary {
      background:#eef4fb; color:#174d91; border:1px solid #bfd2e8; border-radius:6px; padding:9px 12px; min-height:40px; font-weight:750; cursor:pointer;
    }
    .action-row, .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:14px; }
    .monitor-head { padding:22px; border-bottom:1px solid var(--line); background:#fff; }
    .status-strip { display:flex; gap:10px; align-items:center; justify-content:space-between; padding:14px 22px; border-bottom:1px solid var(--line); background:#fbfdff; }
    .monitor-actions { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .badge { border-radius:999px; padding:5px 10px; font-size:12px; font-weight:800; background:#e8eef6; color:#334155; }
    .badge.running { background:#dbeafe; color:#1d4ed8; }
    .badge.cancelling, .badge.cancelled { background:#fef3c7; color:#92400e; }
    .badge.completed { background:#dcfce7; color:#166534; }
    .badge.failed { background:#fee2e2; color:#991b1b; }
    .danger-button { background:#fff5f5 !important; color:#b42318 !important; border-color:#fecaca !important; }
    .progress-row { display:flex; align-items:center; justify-content:space-between; gap:12px; color:var(--muted); margin-bottom:8px; }
    progress { width:100%; height:10px; border:0; border-radius:999px; overflow:hidden; accent-color:var(--accent); }
    .muted { color:var(--muted); }
    table { width:100%; border-collapse:collapse; margin-top:10px; }
    th, td { text-align:left; padding:9px 8px; border-bottom:1px solid #edf1f5; vertical-align:top; }
    th { font-size:12px; color:#53657a; }
    code { background:#eef2f6; padding:2px 5px; border-radius:4px; }
    pre { background:#101722; color:#d6e3f3; border-radius:8px; padding:12px; overflow:auto; min-height:260px; max-height:420px; white-space:pre-wrap; }
    .monitor-table { padding:0 22px 22px; }
    .links { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
    .links a { color:#075f58; font-weight:750; text-decoration:none; }
    .modal-shell {
      display:none; position:fixed; inset:0; z-index:20; padding:24px; background:rgba(15,23,42,.42);
      align-items:center; justify-content:center;
    }
    .modal-shell.open { display:flex; }
    .modal-panel {
      width:min(980px,calc(100vw - 32px)); max-height:calc(100vh - 48px); overflow:hidden;
      background:#fff; border-radius:8px; border:1px solid var(--line); box-shadow:0 24px 60px rgba(15,23,42,.28);
    }
    .modal-head {
      display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 20px; border-bottom:1px solid var(--line);
    }
    .modal-head h2 { margin:4px 0 0; font-size:18px; }
    .api-config-layout { display:grid; grid-template-columns:260px minmax(0,1fr); max-height:calc(100vh - 134px); }
    .api-provider-pane { border-right:1px solid var(--line); background:#fbfdff; padding:16px; overflow:auto; }
    .api-pane-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }
    .provider-list { display:grid; gap:8px; }
    .provider-card {
      min-height:58px; border:1px solid var(--line); border-radius:6px; background:#fff; padding:10px;
      display:grid; grid-template-columns:36px 1fr; align-items:center; gap:10px; text-align:left; cursor:pointer;
    }
    .provider-card.active { border-color:var(--blue); background:#eff6ff; box-shadow:inset 3px 0 0 var(--blue); }
    .provider-icon { width:36px; height:36px; border-radius:8px; display:grid; place-items:center; background:#e8f5f3; color:#075f58; font-weight:850; }
    .provider-card-title { display:block; font-weight:800; color:#172033; }
    .provider-card-desc { display:block; color:var(--muted); font-size:12px; margin-top:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .api-config-main { padding:20px; overflow:auto; }
    .api-provider-summary { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:14px; }
    .provider-summary-left { display:flex; align-items:center; gap:12px; }
    .provider-summary-title { margin:0; font-size:20px; }
    .provider-summary-subtitle { margin:3px 0 0; color:var(--muted); font-size:13px; }
    .model-choice-list { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:8px; max-height:176px; overflow:auto; }
    .model-choice {
      min-height:38px; border:1px solid var(--line); border-radius:6px; background:#fff; padding:8px 10px;
      text-align:left; cursor:pointer; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }
    .model-choice.active { border-color:var(--accent); background:#f0fdfa; box-shadow:inset 3px 0 0 var(--accent); }
    .connection-result { display:none; border:1px solid var(--line); border-radius:6px; padding:10px; margin:12px 0; background:#fbfdff; }
    .connection-result.show { display:block; }
    .connection-result.ok { border-color:#bbf7d0; background:#f0fdf4; color:#166534; }
    .connection-result.error { border-color:#fecaca; background:#fff5f5; color:#991b1b; }
    .ollama-context-panel { display:none; border:1px solid var(--line); border-radius:6px; padding:12px; background:#fbfdff; margin:10px 0; }
    .ollama-context-panel.show { display:block; }
    .context-row { display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:8px; }
    .context-value { font-weight:800; color:var(--blue); }
    .context-slider { width:100%; }
    .context-ticks { display:flex; justify-content:space-between; color:var(--muted); font-size:11px; margin-top:4px; }
    @media (max-width:1060px) {
      .shell { grid-template-columns:1fr; }
      .app-header { align-items:flex-start; flex-direction:column; }
      .top-pills { justify-content:flex-start; }
      .api-config-layout { grid-template-columns:1fr; }
      .api-provider-pane { border-right:0; border-bottom:1px solid var(--line); }
      .provider-list { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    @media (max-width:640px) {
      .shell { padding:14px; }
      .grid, .triple-grid, .workspace-tabs, .metric-strip, .compare-hero, .compare-cards, .secret-grid, .model-choice-list, .provider-list { grid-template-columns:1fr; }
      .panel-pad, .section-head, .monitor-head, .run-body { padding-left:16px; padding-right:16px; }
      .modal-shell { padding:12px; }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="brand">
      <div class="brand-mark">RF</div>
      <div>
        <h1>RAGEval Forge</h1>
        <p>从文档生成可追溯问答集，再用 DeepLocals 真实链路做整集测评。</p>
      </div>
    </div>
    <div class="top-pills">
      <span class="pill"><span class="dot"></span>纯 Node.js</span>
      <span class="pill"><span class="dot"></span>可配置对话模型</span>
      <span class="pill"><span class="dot"></span>全量数据集测评</span>
      <button id="apiSettingsButton" class="secondary" type="button">API 设置</button>
      <button class="secondary" onclick="refreshAll()">刷新</button>
    </div>
  </header>
  <main class="shell">
    <section class="panel workspace">
      <div class="workspace-tabs">
        <button class="workspace-tab active" type="button" data-workspace-tab="build">数据集生成</button>
        <button class="workspace-tab" type="button" data-workspace-tab="eval">真实测评</button>
        <button class="workspace-tab" type="button" data-workspace-tab="compare">平台对比</button>
      </div>

      <div class="workflow-panel active" data-workspace-panel="build">
        <div class="section-head">
          <div class="kicker">Dataset Builder</div>
          <h2>生成问答对与证据</h2>
          <p>上传 PDF 或文档后，系统会用 MinerU 解析全文，再按证据块生成问题、答案和可核验证据。</p>
        </div>
        <form id="job-form" class="panel-pad">
          <div class="metric-strip">
            <div class="metric"><span>LLM 后端</span><strong id="provider-metric">加载中</strong></div>
            <div class="metric"><span>模型</span><strong id="model-metric">加载中</strong></div>
            <div class="metric"><span>负例</span><strong>不生成</strong></div>
          </div>
          <label class="file-drop">
            <span class="file-title">上传原始文档</span>
            <span class="file-subtitle">支持 PDF、DOCX、PPTX、XLSX、图片、HTML、TXT、Markdown。PDF 超过 200 页会自动拆分后送 MinerU。</span>
            <input id="sourceFiles" name="files" type="file" multiple />
          </label>
          <div class="grid" style="margin-top:14px">
            <div><label>数据集名</label><input id="buildName" name="dataset_name" value="custom_rag_eval" /></div>
            <div><label>MinerU 模型</label><select id="mineruModel" name="mineru_model"><option value="vlm">vlm</option><option value="pipeline">pipeline</option></select></div>
            <div><label>目标题数</label><input id="targetQuestions" name="target_count" type="number" value="20" /></div>
            <div><label>每块题数</label><input id="questionsPerChunk" name="questions_per_chunk" type="number" value="3" /></div>
          </div>
          <div class="toggle-grid">
            <label class="checkrow"><input id="useMineru" name="use_mineru" type="checkbox" checked />使用 MinerU 解析</label>
            <label class="checkrow"><input id="isOcr" name="is_ocr" type="checkbox" checked />启用 OCR</label>
            <label class="checkrow"><input id="enableTable" name="enable_table" type="checkbox" checked />表格识别</label>
            <label class="checkrow"><input id="enableFormula" name="enable_formula" type="checkbox" checked />公式识别</label>
          </div>
          <div class="field"><label>可选：直接粘贴文本</label>
            <textarea id="sourceText" placeholder="没有上传文件时，可以粘贴 TXT/Markdown/HTML 文本生成数据集"></textarea>
          </div>
          <div class="action-row"><button class="primary" type="submit">开始生成数据集</button><span class="muted">MinerU Token 使用环境变量或 .mineru_api_key；问答生成使用当前 LLM 配置。</span></div>
        </form>
      </div>

      <div class="workflow-panel" data-workspace-panel="eval">
        <div class="section-head">
          <div class="kicker">Real RAG Evaluation</div>
          <h2>选择整个数据集进行真实测评</h2>
          <p>选中数据集后，上传对应 corpus 到新的 DeepLocals 知识库，并用该数据集问答对完成端到端评估。</p>
        </div>
        <div class="panel-pad">
        <div class="field">
          <label>测评数据集</label>
          <input id="dataset" type="hidden" value="zh_int_clean" />
          <div class="dataset-picker">
            <button id="datasetPickerButton" class="picker-button secondary" type="button" aria-expanded="false">
              <span>
                <span id="pickerTitle" class="picker-title">加载数据集...</span>
                <span id="pickerMeta" class="picker-meta">正在扫描 data/</span>
              </span>
              <span class="chevron">v</span>
            </button>
            <div id="datasetPickerMenu" class="picker-menu">
              <input id="datasetSearch" class="picker-search" type="search" placeholder="搜索数据集名" />
              <div id="datasetList" class="dataset-list"></div>
            </div>
          </div>
        </div>
        <div class="grid">
          <div><label>题目上限，0 为全量</label><input id="limit" type="number" value="0" /></div>
          <div><label>DeepLocals 地址</label><input id="deepBase" value="http://127.0.0.1:3335" /></div>
          <div><label>超时秒数</label><input id="timeout" type="number" value="900" /></div>
        </div>
        <div class="action-row">
          <button class="primary" onclick="runDeepLocals()">开始真实测评</button>
          <span class="muted">回答模型使用环境变量或本地配置。</span>
        </div>
        </div>
      </div>

      <div class="workflow-panel" data-workspace-panel="compare">
        <div class="section-head">
          <div class="kicker">Platform Comparison</div>
          <h2>DeepLocals vs Cherry Studio</h2>
          <p>独立的端到端平台对比入口。两边会并发执行，DeepLocals 和 Cherry 各自走自己的平台能力。</p>
        </div>
        <div class="panel-pad">
          <div class="compare-hero">
            <div>
              <div class="kicker">Ready To Run</div>
              <h2>并发端到端平台对比</h2>
              <p class="muted">默认固定 Cherry 检索 20 块，DeepLocals 使用同一数据集和同一评分规则。</p>
            </div>
            <button class="primary" onclick="runCompare()">并发平台对比</button>
          </div>
          <div class="compare-cards">
            <div class="compare-card"><span>DeepLocals</span><strong id="deepSummary">http://127.0.0.1:3335</strong></div>
            <div class="compare-card"><span>Cherry Studio</span><strong id="cherrySummary">http://127.0.0.1:23333</strong></div>
          </div>
          <div class="grid" style="margin-top:14px">
            <div><label>测评数据集</label><select id="compareDataset"></select></div>
            <div><label>题目上限，0 为全量</label><input id="compareLimit" type="number" value="0" /></div>
            <div><label>DeepLocals 地址</label><input id="compareDeepBase" value="http://127.0.0.1:3335" /></div>
            <div><label>DeepLocals 模型服务</label><input id="deepProvider" value="siliconflow" /></div>
            <div><label>DeepLocals 回答模型</label><input id="deepModel" value="deepseek-ai/DeepSeek-V4-Flash" /></div>
            <div><label>DeepLocals API 地址</label><input id="deepApiUrl" value="https://api.siliconflow.cn/v1" /></div>
            <div><label>Cherry API 地址</label><input id="cherryBase" value="http://127.0.0.1:23333" /></div>
            <div><label>Cherry Knowledge Base ID</label><input id="cherryKb" /></div>
            <div><label>Cherry 模型</label><input id="cherryModel" value="silicon:deepseek-ai/DeepSeek-V4-Flash" /></div>
            <div><label>Cherry 检索块数</label><input id="documentCount" type="number" value="20" /></div>
          </div>
          <div class="secret-grid" style="margin-top:14px">
            <div><label>Cherry API Key，可留空使用环境变量</label><input id="cherryKey" type="password" /></div>
            <div><label>DeepLocals 回答模型 Key，可留空使用环境变量</label><input id="deepKey" type="password" /></div>
          </div>
          <div class="action-row">
            <button class="secondary" onclick="runCherry()">只测 Cherry</button>
          </div>
        </div>
      </div>
    </section>

    <aside class="panel monitor">
      <div class="monitor-head">
        <div class="kicker">Run Monitor</div>
        <h2>任务状态</h2>
        <p>所有生成、真实测评和平台对比任务都会在这里显示进度与报告路径。</p>
      </div>
      <div class="status-strip">
        <div>
          <div id="jobTitle" class="muted">尚未开始</div>
          <strong id="jobId">idle</strong>
        </div>
        <div class="monitor-actions">
          <button id="cancelJobButton" class="secondary danger-button" type="button" disabled>取消任务</button>
          <span id="badge" class="badge">idle</span>
        </div>
      </div>
      <div class="run-body">
        <div class="progress-row"><span>当前进度</span><span id="progressText">0%</span></div>
        <progress id="progressBar" value="0" max="100"></progress>
        <pre id="log">等待任务...</pre>
        <div id="links" class="links"></div>
      </div>
      <div class="monitor-table">
        <h2>数据集</h2>
        <table id="datasets"></table>
        <h2 style="margin-top:22px">最近报告</h2>
        <table id="reports"></table>
      </div>
    </aside>
  </main>
  <div id="apiSettingsModal" class="modal-shell" aria-hidden="true">
    <div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="apiSettingsTitle">
      <div class="modal-head">
        <div>
          <div class="kicker">API Settings</div>
          <h2 id="apiSettingsTitle">对话模型 API 配置</h2>
        </div>
        <button id="apiSettingsClose" class="secondary" type="button" aria-label="关闭 API 设置">×</button>
      </div>
      <form id="llmConfigForm" class="api-config-layout">
        <aside class="api-provider-pane">
          <div class="api-pane-head">
            <strong>选择提供商</strong>
            <button id="customProviderButton" class="secondary" type="button">自定义</button>
          </div>
          <input id="llmProvider" name="provider_id" type="hidden" value="mimo" />
          <div id="providerList" class="provider-list"></div>
        </aside>
        <section class="api-config-main">
          <div class="api-provider-summary">
            <div class="provider-summary-left">
              <span id="providerSummaryIcon" class="provider-icon">AI</span>
              <div>
                <h3 id="providerSummaryTitle" class="provider-summary-title">加载中</h3>
                <p id="providerSummarySubtitle" class="provider-summary-subtitle">OpenAI-compatible 对话模型接口</p>
              </div>
            </div>
            <label class="checkrow" style="margin:0"><input id="llmEnabled" name="enabled" type="checkbox" checked />启用</label>
          </div>
          <div class="grid">
            <div class="field"><label>API Key</label><input id="llmApiKey" name="api_key" type="password" autocomplete="off" placeholder="留空则继续使用已保存或本地 key 文件" /></div>
            <div class="field"><label>API 地址</label><input id="llmApiUrl" name="api_url" value="https://api.siliconflow.cn/v1" /></div>
          </div>
          <div id="ollamaContextPanel" class="ollama-context-panel">
            <div class="context-row"><label>Ollama 上下文长度</label><span id="ollamaContextValue" class="context-value">8K</span></div>
            <input id="ollamaContextSlider" class="context-slider" type="range" min="0" max="6" step="1" value="1" />
            <div class="context-ticks"><span>4K</span><span>8K</span><span>16K</span><span>32K</span><span>64K</span><span>128K</span><span>256K</span></div>
          </div>
          <div id="llmConnectionStatus" class="connection-result" aria-live="polite">
            <strong id="llmConnectionTitle">连接状态</strong>
            <div id="llmConnectionDetail">等待测试连接。</div>
          </div>
          <input id="llmModel" name="model" type="hidden" value="" />
          <div class="field">
            <label>模型列表</label>
            <div class="muted" id="modelChoiceHint">测试连接后可刷新列表，也可直接保存当前模型。</div>
            <div id="modelChoiceList" class="model-choice-list"></div>
          </div>
          <div class="grid">
            <div class="field"><label>鉴权方式</label><select id="llmAuthType" name="auth_type"><option value="bearer">Authorization: Bearer</option><option value="api-key">api-key</option></select></div>
            <div class="field"><label>最大输出 Tokens</label><input id="llmMaxTokens" name="max_tokens" type="number" min="1" max="32768" value="4096" /></div>
            <div class="field"><label>Temperature</label><input id="llmTemperature" name="temperature" type="number" min="0" max="2" step="0.1" value="0" /></div>
            <div class="field"><label>Top P</label><input id="llmTopP" name="top_p" type="number" min="0.1" max="1" step="0.05" value="0.95" /></div>
          </div>
          <label class="checkrow"><input id="llmEnableThinking" name="enable_thinking" type="checkbox" />启用 SiliconFlow Qwen thinking</label>
          <div class="action-row">
            <button id="llmTestButton" class="secondary" type="button">测试连接</button>
            <span id="llmConfigStatus" class="muted">正在加载当前配置...</span>
            <button id="llmSaveButton" class="primary" type="submit">保存配置</button>
          </div>
        </section>
      </form>
    </div>
  </div>
  <script>
    const state = {
      jobs: new Map(),
      activeTab: 'build',
      activeJobId: null,
      datasets: [],
      selectedDataset: '',
      providerPresets: [],
      llmConfig: null,
      modelList: [],
      ollamaContextLength: 8192
    };
    const savedKeyPlaceholder = '__RAGEVAL_SAVED_API_KEY__';
    const ollamaContextOptions = [4096, 8192, 16384, 32768, 65536, 131072, 262144];
    document.querySelectorAll('[data-workspace-tab]').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.workspaceTab)));
    function switchTab(tab) {
      state.activeTab = tab;
      document.querySelectorAll('[data-workspace-tab]').forEach(btn => btn.classList.toggle('active', btn.dataset.workspaceTab === tab));
      document.querySelectorAll('[data-workspace-panel]').forEach(sec => sec.classList.toggle('active', sec.dataset.workspacePanel === tab));
    }
    async function api(path, options = {}) {
      const res = await fetch(path, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    }
    function selectedPreset(providerId) {
      return state.providerPresets.find(p => p.id === providerId) || state.providerPresets[0] || null;
    }
    function providerVisual(providerId) {
      const visuals = {
        mimo: ['MM', 'mimo'],
        siliconflow: ['SF', 'siliconflow'],
        deepseek: ['DS', 'deepseek'],
        qwen: ['QW', 'qwen'],
        ollama: ['OL', 'ollama'],
        custom: ['+', 'custom']
      };
      return visuals[providerId] || ['AI', 'custom'];
    }
    function providerDescription(preset) {
      const descriptions = {
        mimo: 'MiMo 对话模型服务',
        siliconflow: '高性能 AI 推理服务平台',
        deepseek: 'DeepSeek 官方兼容接口',
        qwen: '阿里云百炼兼容接口',
        ollama: '本地运行的 Ollama 模型',
        custom: '兼容 OpenAI Chat Completions'
      };
      return descriptions[preset?.id] || preset?.description || 'OpenAI-compatible 接口';
    }
    function presetToConfig(preset) {
      return {
        provider_id: preset.id,
        provider_name: preset.name,
        api_key: '',
        api_url: preset.api_url,
        model: preset.default_model,
        models: preset.models || [preset.default_model],
        auth_type: preset.auth_type || 'bearer',
        max_tokens: preset.max_tokens || 4096,
        temperature: preset.temperature ?? 0,
        top_p: preset.top_p ?? 0.95,
        enabled: true,
        extra_body: preset.extra_body || {}
      };
    }
    function renderProviderList(activeId) {
      const list = document.getElementById('providerList');
      list.innerHTML = state.providerPresets.map(preset => {
        const visual = providerVisual(preset.id);
        return '<button class="provider-card' + (preset.id === activeId ? ' active' : '') + '" type="button" data-provider="' + escapeHtml(preset.id) + '">' +
          '<span class="provider-icon">' + escapeHtml(visual[0]) + '</span>' +
          '<span><span class="provider-card-title">' + escapeHtml(preset.name) + '</span><span class="provider-card-desc">' + escapeHtml(providerDescription(preset)) + '</span></span>' +
        '</button>';
      }).join('');
      list.querySelectorAll('[data-provider]').forEach(button => {
        button.addEventListener('click', () => {
          const preset = selectedPreset(button.dataset.provider);
          if (preset) renderLlmConfig(presetToConfig(preset), false);
        });
      });
    }
    function renderModelChoices(models, activeModel) {
      const list = document.getElementById('modelChoiceList');
      const values = (models || []).filter(Boolean);
      if (!values.length) {
        list.innerHTML = '<div class="empty-state">暂无模型列表。可以直接保存当前模型名，或先测试连接。</div>';
        return;
      }
      list.innerHTML = values.map(model => '<button class="model-choice' + (model === activeModel ? ' active' : '') + '" type="button" data-model="' + escapeHtml(model) + '">' + escapeHtml(model) + '</button>').join('');
      list.querySelectorAll('[data-model]').forEach(button => {
        button.addEventListener('click', () => {
          document.getElementById('llmModel').value = button.dataset.model;
          renderModelChoices(values, button.dataset.model);
        });
      });
    }
    function setConnectionStatus(type, title, detail) {
      const box = document.getElementById('llmConnectionStatus');
      box.className = 'connection-result';
      if (!type) return;
      box.classList.add('show', type);
      document.getElementById('llmConnectionTitle').textContent = title || '';
      document.getElementById('llmConnectionDetail').textContent = detail || '';
    }
    function contextLabel(value) {
      return Math.round(Number(value || 8192) / 1024) + 'K';
    }
    function renderOllamaContextLength(value) {
      state.ollamaContextLength = Number(value || 8192);
      const index = Math.max(0, ollamaContextOptions.indexOf(state.ollamaContextLength));
      document.getElementById('ollamaContextSlider').value = String(index);
      document.getElementById('ollamaContextValue').textContent = contextLabel(state.ollamaContextLength);
    }
    function updateProviderMode(providerId) {
      const isOllama = providerId === 'ollama';
      document.getElementById('llmApiKey').disabled = isOllama;
      document.getElementById('llmApiKey').placeholder = isOllama ? 'Ollama 本地无需密钥' : '留空则继续使用已保存或本地 key 文件';
      document.getElementById('ollamaContextPanel').classList.toggle('show', isOllama);
      document.getElementById('llmTestButton').textContent = isOllama ? '刷新本地模型' : '测试连接';
    }
    function renderLlmConfig(config, fromServer = true) {
      state.llmConfig = config;
      state.modelList = config.models || [];
      const providerId = config.provider_id || 'mimo';
      const preset = selectedPreset(providerId) || { name: config.provider_name || providerId, id: providerId };
      const visual = providerVisual(providerId);
      document.getElementById('llmProvider').value = providerId;
      document.getElementById('providerSummaryIcon').textContent = visual[0];
      document.getElementById('providerSummaryTitle').textContent = config.provider_name || preset.name || providerId;
      document.getElementById('providerSummarySubtitle').textContent = providerDescription(preset);
      document.getElementById('llmEnabled').checked = config.enabled !== false;
      document.getElementById('llmApiUrl').value = config.api_url || '';
      document.getElementById('llmModel').value = config.model || '';
      document.getElementById('llmAuthType').value = config.auth_type || 'bearer';
      document.getElementById('llmMaxTokens').value = config.max_tokens || 4096;
      document.getElementById('llmTemperature').value = config.temperature ?? 0;
      document.getElementById('llmTopP').value = config.top_p ?? 0.95;
      document.getElementById('llmEnableThinking').checked = Boolean(config.extra_body && config.extra_body.enable_thinking);
      const keyInput = document.getElementById('llmApiKey');
      if (providerId === 'ollama') {
        keyInput.value = '';
      } else if (fromServer && config.api_key_masked) {
        keyInput.value = savedKeyPlaceholder;
        keyInput.placeholder = '已保存 key：' + config.api_key_masked;
      } else {
        keyInput.value = config.api_key || '';
      }
      updateProviderMode(providerId);
      renderProviderList(providerId);
      renderModelChoices(state.modelList, config.model);
      document.getElementById('provider-metric').textContent = config.provider_name || providerId;
      document.getElementById('model-metric').textContent = config.model || '-';
    }
    function buildLlmConfigPayload() {
      const apiKey = document.getElementById('llmApiKey').value === savedKeyPlaceholder ? '' : document.getElementById('llmApiKey').value;
      return {
        provider_id: val('llmProvider'),
        provider_name: document.getElementById('providerSummaryTitle').textContent,
        enabled: checked('llmEnabled'),
        api_key: apiKey,
        api_url: val('llmApiUrl'),
        model: val('llmModel'),
        models: state.modelList,
        auth_type: val('llmAuthType'),
        max_tokens: Number(val('llmMaxTokens') || 4096),
        temperature: Number(val('llmTemperature') || 0),
        top_p: Number(val('llmTopP') || 0.95),
        enable_thinking: checked('llmEnableThinking')
      };
    }
    async function loadLlmSettings() {
      const payload = await api('/api/llm-config');
      state.providerPresets = payload.presets || [];
      renderLlmConfig(payload.config || presetToConfig(state.providerPresets[0]), true);
      const context = await api('/api/ollama/context-config').catch(() => ({ context_length: 8192 }));
      renderOllamaContextLength(context.context_length || 8192);
      document.getElementById('llmConfigStatus').textContent = '当前配置已加载';
    }
    function openApiSettings() {
      document.getElementById('apiSettingsModal').classList.add('open');
      document.getElementById('apiSettingsModal').setAttribute('aria-hidden', 'false');
    }
    function closeApiSettings() {
      document.getElementById('apiSettingsModal').classList.remove('open');
      document.getElementById('apiSettingsModal').setAttribute('aria-hidden', 'true');
    }
    async function testLlmConnection() {
      setConnectionStatus('', '', '');
      document.getElementById('llmConfigStatus').textContent = '正在测试连接...';
      try {
        const payload = await api('/api/llm-config/test', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify(buildLlmConfigPayload())
        });
        state.modelList = payload.models || [];
        renderLlmConfig(payload.config || { ...buildLlmConfigPayload(), models: state.modelList }, true);
        setConnectionStatus('ok', '连接成功', payload.message || '模型列表已刷新');
        document.getElementById('llmConfigStatus').textContent = '连接可用';
      } catch (error) {
        setConnectionStatus('error', '连接失败', error.message || String(error));
        document.getElementById('llmConfigStatus').textContent = '连接失败';
      }
    }
    async function saveLlmSettings() {
      document.getElementById('llmConfigStatus').textContent = '正在保存...';
      const providerId = val('llmProvider');
      if (providerId === 'ollama') {
        await api('/api/ollama/context-config', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({ context_length: state.ollamaContextLength })
        }).catch(() => null);
      }
      const payload = await api('/api/llm-config', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(buildLlmConfigPayload())
      });
      state.providerPresets = payload.presets || state.providerPresets;
      renderLlmConfig(payload.config || state.llmConfig, true);
      document.getElementById('llmConfigStatus').textContent = '已保存';
    }
    async function refreshAll(preferredDataset = '') { await Promise.all([refreshDatasets(preferredDataset), refreshReports()]); renderJobs(); }
    async function refreshDatasets(preferredDataset = '') {
      const { datasets } = await api('/api/datasets');
      state.datasets = datasets;
      const datasetTable = document.getElementById('datasets');
      datasetTable.innerHTML = '<tr><th>名称</th><th>题数</th><th>Corpus</th><th>更新时间</th><th></th></tr>' + datasets.map(d =>
        '<tr><td><code>' + escapeHtml(d.name) + '</code></td><td>' + d.rows + '</td><td>' + (d.has_corpus ? '有' : '无') + '</td><td>' + new Date(d.updated_at).toLocaleString() + '</td><td><button class="icon-danger" type="button" title="删除数据集" aria-label="删除 ' + escapeHtml(d.name) + '" data-delete-dataset="' + escapeHtml(d.name) + '">×</button></td></tr>'
      ).join('');
      datasetTable.querySelectorAll('[data-delete-dataset]').forEach(button => {
        button.addEventListener('click', () => deleteDataset(button.dataset.deleteDataset));
      });
      const available = datasets.filter(d => d.has_corpus);
      const wanted = preferredDataset || state.selectedDataset || localStorage.getItem('rageval:selectedDataset') || valOrDefault('buildName', 'zh_int_clean');
      const selected = available.find(d => d.name === wanted) || available[0] || datasets[0] || null;
      setSelectedDataset(selected);
      renderDatasetList(valOrDefault('datasetSearch', ''));
      renderCompareDatasetSelect();
    }
    async function refreshReports() {
      const { reports } = await api('/api/reports');
      document.getElementById('reports').innerHTML = '<tr><th>报告</th><th>数据集</th><th>准确率</th><th>正确</th></tr>' + reports.map(r => {
        const acc = r.qa_accuracy == null ? '-' : (r.qa_accuracy * 100).toFixed(2) + '%';
        return '<tr><td><code>' + escapeHtml(r.display_name || r.name) + '</code></td><td>' + escapeHtml(r.dataset || '-') + '</td><td>' + acc + '</td><td>' + (r.qa_correct ?? '-') + '/' + (r.qa_total ?? '-') + '</td></tr>';
      }).join('');
    }
    function setSelectedDataset(item) {
      const datasetInput = document.getElementById('dataset');
      const title = document.getElementById('pickerTitle');
      const meta = document.getElementById('pickerMeta');
      if (!item || !item.has_corpus) {
        state.selectedDataset = '';
        datasetInput.value = '';
        title.textContent = item ? item.name : '暂无可测数据集';
        meta.textContent = item ? '缺少 corpus，需先重新生成或补齐原始文档' : '先生成一个带 corpus 的数据集';
        return;
      }
      state.selectedDataset = item.name;
      datasetInput.value = item.name;
      localStorage.setItem('rageval:selectedDataset', item.name);
      title.textContent = item.name;
      meta.textContent = item.rows + ' 题 · ' + (item.has_assets ? '有原始资产' : '有 corpus') + ' · ' + (item.corpus_path || '');
      const compare = document.getElementById('compareDataset');
      if (compare) compare.value = item.name;
    }
    function renderDatasetList(needle = '') {
      const box = document.getElementById('datasetList');
      if (!box) return;
      const query = String(needle || '').trim().toLowerCase();
      const visible = state.datasets.filter(d => d.name.toLowerCase().includes(query));
      if (!visible.length) {
        box.innerHTML = '<div class="empty-state">没有匹配的数据集。</div>';
        return;
      }
      box.innerHTML = visible.map(d => {
        const tagClass = d.has_corpus ? 'tag' : 'tag warn';
        const tag = d.has_corpus ? '可测' : '缺 corpus';
        const cls = 'dataset-option' + (d.name === state.selectedDataset ? ' selected' : '') + (!d.has_corpus ? ' unavailable' : '');
        return '<div class="' + cls + '" data-name="' + escapeHtml(d.name) + '">' +
          '<button class="dataset-main" type="button" ' + (d.has_corpus ? '' : 'disabled') + '>' +
            '<span class="dataset-name">' + escapeHtml(d.name) + '</span>' +
            '<span class="dataset-meta">' + d.rows + ' 题 · ' + escapeHtml(d.has_assets ? '有原始资产' : d.has_corpus ? '有 corpus' : '需生成 corpus') + '</span>' +
          '</button>' +
          '<span class="dataset-actions">' +
            '<span class="' + tagClass + '">' + tag + '</span>' +
            '<button class="icon-danger" type="button" title="删除数据集" aria-label="删除 ' + escapeHtml(d.name) + '" data-delete-dataset="' + escapeHtml(d.name) + '">×</button>' +
          '</span>' +
        '</div>';
      }).join('');
      box.querySelectorAll('.dataset-option').forEach(option => {
        option.querySelector('.dataset-main').addEventListener('click', () => {
          const item = state.datasets.find(d => d.name === option.dataset.name);
          if (!item || !item.has_corpus) return;
          setSelectedDataset(item);
          renderDatasetList(document.getElementById('datasetSearch').value);
          closeDatasetPicker();
        });
        option.querySelector('[data-delete-dataset]').addEventListener('click', (event) => {
          event.stopPropagation();
          deleteDataset(event.currentTarget.dataset.deleteDataset);
        });
      });
    }
    function renderCompareDatasetSelect() {
      const select = document.getElementById('compareDataset');
      if (!select) return;
      const available = state.datasets.filter(d => d.has_corpus);
      select.innerHTML = available.length
        ? available.map(d => '<option value="' + escapeHtml(d.name) + '">' + escapeHtml(d.name) + ' · ' + d.rows + ' 题</option>').join('')
        : '<option value="">暂无可测数据集</option>';
      select.disabled = !available.length;
      if (state.selectedDataset) select.value = state.selectedDataset;
    }
    function closeDatasetPicker() {
      const menu = document.getElementById('datasetPickerMenu');
      const button = document.getElementById('datasetPickerButton');
      if (menu) menu.classList.remove('open');
      if (button) button.setAttribute('aria-expanded', 'false');
    }
    async function deleteDataset(name) {
      if (!name) return;
      if (!window.confirm('删除数据集「' + name + '」？\n会同时删除 JSON、corpus、报告和 assets 目录。')) return;
      try {
        const payload = await api('/api/datasets/' + encodeURIComponent(name), { method: 'DELETE' });
        if (state.selectedDataset === name) {
          state.selectedDataset = '';
          localStorage.removeItem('rageval:selectedDataset');
        }
        document.getElementById('badge').className = 'badge completed';
        document.getElementById('badge').textContent = 'completed';
        document.getElementById('log').textContent = '已删除数据集：' + name + '\n' + (payload.deleted || []).join('\n');
        await refreshAll();
      } catch (error) {
        document.getElementById('badge').className = 'badge failed';
        document.getElementById('badge').textContent = 'failed';
        document.getElementById('log').textContent = error.message || String(error);
      }
    }
    async function startJob(path, payload) {
      const isForm = payload instanceof FormData;
      const { job } = await api(path, isForm
        ? { method:'POST', body: payload }
        : { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      state.jobs.set(job.id, job);
      state.activeJobId = job.id;
      renderJobs();
      pollJob(job.id);
    }
    async function pollJob(id) {
      const timer = setInterval(async () => {
        const { job } = await api('/api/jobs/' + encodeURIComponent(id)).catch(err => ({ job: { id, status:'failed', error:err.message, logs:[] } }));
        state.jobs.set(id, job);
        renderJobs();
        if (job.status !== 'running' && job.status !== 'cancelling') {
          clearInterval(timer);
          const preferred = job.kind === 'build_dataset' && job.status === 'completed' ? valOrDefault('buildName', '') : '';
          refreshAll(preferred);
        }
      }, 1500);
    }
    function renderJobs() {
      const jobs = [...state.jobs.values()].sort((a,b) => b.created_at.localeCompare(a.created_at));
      const job = state.activeJobId ? state.jobs.get(state.activeJobId) : jobs[0];
      const badge = document.getElementById('badge');
      badge.className = 'badge' + (job ? ' ' + job.status : '');
      badge.textContent = job ? job.status : 'idle';
      document.getElementById('jobTitle').textContent = job ? job.kind : '尚未开始';
      document.getElementById('jobId').textContent = job ? job.id : 'idle';
      const progress = job ? Number(job.progress || 0) : 0;
      document.getElementById('progressBar').value = progress;
      document.getElementById('progressText').textContent = Math.round(progress) + '%';
      document.getElementById('cancelJobButton').disabled = !job || job.status !== 'running';
      const logs = job ? (job.logs || []).map(x => x.message).join('\n') : '';
      document.getElementById('log').textContent = job?.error || logs || '等待任务...';
      document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
      const linkItems = [];
      if (job?.output_url) linkItems.push('<a href="' + escapeHtml(job.output_url) + '">下载输出</a>');
      if (job?.report_url) linkItems.push('<a href="' + escapeHtml(job.report_url) + '">下载报告</a>');
      if (job?.summary_url && job.summary_url !== job.report_url) linkItems.push('<a href="' + escapeHtml(job.summary_url) + '">下载摘要</a>');
      if (job?.csv_url) linkItems.push('<a href="' + escapeHtml(job.csv_url) + '">下载 CSV</a>');
      if (job?.corpus_url) linkItems.push('<a href="' + escapeHtml(job.corpus_url) + '">下载 corpus</a>');
      if (job?.assets_manifest) linkItems.push('<a href="/download?path=' + encodeURIComponent(job.assets_manifest) + '">资产 Manifest</a>');
      document.getElementById('links').innerHTML = linkItems.join('');
    }
    function evalPayload() {
      return {
        dataset: val('dataset'),
        limit: Number(valOrDefault('compareLimit', valOrDefault('limit', '0')) || 0),
        timeout: Number(val('timeout') || 900),
        deeplocalApiBase: valOrDefault('compareDeepBase', val('deepBase')),
        cherryApiBase: val('cherryBase'),
        cherryKnowledgeBaseId: val('cherryKb'),
        cherryModel: val('cherryModel'), documentCount: Number(val('documentCount') || 20),
        cherryApiKey: val('cherryKey'),
        deeplocalAnswerProvider: valOrDefault('deepProvider', 'siliconflow'),
        deeplocalAnswerModel: valOrDefault('deepModel', 'deepseek-ai/DeepSeek-V4-Flash'),
        deeplocalAnswerApiUrl: valOrDefault('deepApiUrl', ''),
        deeplocalAnswerApiKey: val('deepKey')
      };
    }
    function runCompare() { startJob('/api/eval/compare', evalPayload()); }
    function runDeepLocals() { const p = evalPayload(); startJob('/api/eval/deeplocals', { dataset:p.dataset, apiBase:val('deepBase'), limit:Number(val('limit') || 0), timeout:p.timeout }); }
    function runCherry() { const p = evalPayload(); startJob('/api/eval/cherry', { dataset:p.dataset, apiBase:p.cherryApiBase, apiKey:p.cherryApiKey, knowledgeBaseId:p.cherryKnowledgeBaseId, model:p.cherryModel, documentCount:p.documentCount, limit:p.limit, timeout:p.timeout }); }
    function updateCompareSummary() {
      const deepText = valOrDefault('deepModel', 'deepseek-ai/DeepSeek-V4-Flash') + ' · ' + valOrDefault('deepProvider', 'siliconflow') + ' · ' + valOrDefault('compareDeepBase', 'http://127.0.0.1:3335');
      const cherryText = valOrDefault('cherryModel', 'silicon:deepseek-ai/DeepSeek-V4-Flash') + ' · Top ' + valOrDefault('documentCount', '20') + ' · ' + valOrDefault('cherryBase', 'http://127.0.0.1:23333');
      document.getElementById('deepSummary').textContent = deepText;
      document.getElementById('cherrySummary').textContent = cherryText;
    }
    function buildDataset() {
      const files = document.getElementById('sourceFiles').files;
      if (files.length) {
        const form = new FormData(document.getElementById('job-form'));
        form.set('dataset_name', val('buildName'));
        form.set('mineru_model', val('mineruModel'));
        form.set('target_count', String(Number(val('targetQuestions') || 20)));
        form.set('questions_per_chunk', String(Number(val('questionsPerChunk') || 3)));
        form.set('use_mineru', checked('useMineru') ? 'true' : 'false');
        form.set('is_ocr', checked('isOcr') ? 'true' : 'false');
        form.set('enable_table', checked('enableTable') ? 'true' : 'false');
        form.set('enable_formula', checked('enableFormula') ? 'true' : 'false');
        startJob('/api/datasets/build', form);
        return;
      }
      const text = val('sourceText');
      if (!text) {
        document.getElementById('badge').className = 'badge failed';
        document.getElementById('badge').textContent = 'failed';
        document.getElementById('log').textContent = '请先上传文件，或在文本框中粘贴原文。';
        return;
      }
      startJob('/api/datasets/build', {
        datasetName: val('buildName'), sourceName: val('buildName') + '.txt', targetQuestions: Number(val('targetQuestions') || 20),
        questionsPerChunk: Number(val('questionsPerChunk') || 3), text
      });
    }
    document.getElementById('job-form').addEventListener('submit', (event) => { event.preventDefault(); buildDataset(); });
    document.getElementById('datasetPickerButton').addEventListener('click', () => {
      const menu = document.getElementById('datasetPickerMenu');
      const open = !menu.classList.contains('open');
      menu.classList.toggle('open', open);
      document.getElementById('datasetPickerButton').setAttribute('aria-expanded', String(open));
      if (open) {
        document.getElementById('datasetSearch').focus();
        renderDatasetList(document.getElementById('datasetSearch').value);
      }
    });
    document.getElementById('datasetSearch').addEventListener('input', (event) => renderDatasetList(event.target.value));
    document.getElementById('compareDataset').addEventListener('change', (event) => {
      const item = state.datasets.find(d => d.name === event.target.value);
      if (item) {
        setSelectedDataset(item);
        renderDatasetList(document.getElementById('datasetSearch').value);
      }
    });
    ['compareDeepBase','deepProvider','deepModel','deepApiUrl','cherryBase','cherryModel','documentCount'].forEach(id => {
      const element = document.getElementById(id);
      if (element) element.addEventListener('input', updateCompareSummary);
    });
    document.addEventListener('click', (event) => {
      const picker = document.querySelector('.dataset-picker');
      if (picker && !picker.contains(event.target)) closeDatasetPicker();
    });
    document.getElementById('apiSettingsButton').addEventListener('click', openApiSettings);
    document.getElementById('apiSettingsClose').addEventListener('click', closeApiSettings);
    document.getElementById('apiSettingsModal').addEventListener('click', (event) => {
      if (event.target.id === 'apiSettingsModal') closeApiSettings();
    });
    document.getElementById('customProviderButton').addEventListener('click', () => {
      const preset = selectedPreset('custom');
      if (preset) renderLlmConfig(presetToConfig(preset), false);
    });
    document.getElementById('llmTestButton').addEventListener('click', testLlmConnection);
    document.getElementById('llmConfigForm').addEventListener('submit', async (event) => {
      event.preventDefault();
      try {
        await saveLlmSettings();
      } catch (error) {
        document.getElementById('llmConfigStatus').textContent = '保存失败：' + (error.message || String(error));
        setConnectionStatus('error', '保存失败', error.message || String(error));
      }
    });
    document.getElementById('ollamaContextSlider').addEventListener('input', (event) => {
      const index = Number(event.target.value || 1);
      renderOllamaContextLength(ollamaContextOptions[index] || 8192);
    });
    document.getElementById('cancelJobButton').addEventListener('click', async () => {
      const id = state.activeJobId;
      if (!id) return;
      document.getElementById('cancelJobButton').disabled = true;
      await api('/api/jobs/' + encodeURIComponent(id) + '/cancel', { method:'POST' }).catch(() => null);
      const { job } = await api('/api/jobs/' + encodeURIComponent(id)).catch(() => ({ job: null }));
      if (job) {
        state.jobs.set(id, job);
        renderJobs();
      }
    });
    function val(id) { return document.getElementById(id).value.trim(); }
    function valOrDefault(id, fallback = '') {
      const element = document.getElementById(id);
      return element && 'value' in element ? element.value.trim() : fallback;
    }
    function checked(id) { return Boolean(document.getElementById(id).checked); }
    function escapeHtml(s) { return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    updateCompareSummary();
    refreshAll();
    loadLlmSettings().catch(error => {
      document.getElementById('llmConfigStatus').textContent = '配置加载失败：' + (error.message || String(error));
    });
  </script>
</body>
</html>`;
}
