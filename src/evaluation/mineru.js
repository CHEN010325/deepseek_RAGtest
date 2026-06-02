import fs from "node:fs/promises";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { Buffer } from "node:buffer";
import { unzipSync } from "fflate";
import { PDFDocument } from "pdf-lib";
import { MINERU_KEY_FILE } from "../config.js";
import { ensureDir, pathExists, readText, sanitizeFilename, uniquePath, writeJson } from "../utils/files.js";

export const MINERU_API_BASE = "https://mineru.net/api/v4";
export const MAX_PAGES_PER_TASK = 200;
const MAX_FILES_PER_BATCH = 50;
const MAX_UPLOAD_BATCH_BYTES = 256 * 1024 * 1024;
const POLL_INTERVAL_MS = 3000;
const POLL_TIMEOUT_MS = 30 * 60 * 1000;

export const MINERU_SUPPORTED_EXTENSIONS = new Set([
  ".pdf",
  ".doc",
  ".docx",
  ".ppt",
  ".pptx",
  ".xls",
  ".xlsx",
  ".png",
  ".jpg",
  ".jpeg",
  ".jp2",
  ".webp",
  ".gif",
  ".bmp",
  ".html",
  ".htm"
]);

const QUOTA_ERROR_PATTERNS = [
  "insufficient balance",
  "insufficient_quota",
  "quota exceeded",
  "quota_exceeded",
  "quota exhausted",
  "out of credits",
  "billing",
  "payment required",
  "resource exhausted",
  "request limit",
  "余额不足",
  "额度不足",
  "配额不足",
  "账户余额",
  "资源耗尽",
  "欠费"
];

export class MineruError extends Error {
  constructor(message) {
    super(message);
    this.name = "MineruError";
  }
}

export class MineruQuotaExceededError extends MineruError {
  constructor(message) {
    super(message);
    this.name = "MineruQuotaExceededError";
  }
}

export async function readMineruToken(rawToken = "") {
  const token = String(rawToken || process.env.MINERU_API_TOKEN || "").trim();
  if (token) return token;
  if (await pathExists(MINERU_KEY_FILE)) {
    const fileToken = (await readText(MINERU_KEY_FILE)).trim();
    if (fileToken) return fileToken;
  }
  throw new MineruError(`MinerU API token is missing. Set MINERU_API_TOKEN or create ${MINERU_KEY_FILE}`);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isQuotaError(statusCode, body) {
  const text = String(body || "").toLowerCase();
  return statusCode === 402 || statusCode === 429 || QUOTA_ERROR_PATTERNS.some((pattern) => text.includes(pattern));
}

async function readResponseText(response) {
  return (await response.text()).slice(0, 1600).replace(/\n/gu, " ");
}

async function assertOk(response, provider) {
  if (response.status < 400) return;
  const body = await readResponseText(response);
  if (isQuotaError(response.status, body)) {
    throw new MineruQuotaExceededError(`${provider} quota or request limit exhausted: HTTP ${response.status} ${body}`);
  }
  throw new MineruError(`${provider} failed: HTTP ${response.status} ${body}`);
}

async function mineruJson(method, apiBase, token, route, payload = null, timeout = 120) {
  const response = await fetch(`${String(apiBase || MINERU_API_BASE).replace(/\/+$/u, "")}${route}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      Accept: "*/*"
    },
    body: payload ? JSON.stringify(payload) : null,
    signal: AbortSignal.timeout(timeout * 1000)
  });
  await assertOk(response, `MinerU ${route}`);
  const data = await response.json();
  if (data?.code !== 0) {
    const message = String(data?.msg || data?.message || JSON.stringify(data)).slice(0, 1200);
    if (isQuotaError(400, message)) throw new MineruQuotaExceededError(`MinerU API rejected request: ${message}`);
    throw new MineruError(`MinerU API rejected request: ${message}`);
  }
  return data;
}

function safeMineruFilename(filename) {
  if (!/[^\x00-\x7F]/u.test(filename)) return filename;
  const ext = path.extname(filename);
  const safeExt = /^[A-Za-z0-9.]+$/u.test(ext) ? ext : ".bin";
  const encoded = Buffer.from(filename, "utf8")
    .toString("base64")
    .replace(/[/+=]/gu, "_")
    .slice(0, 180);
  return `${encoded}${safeExt}`;
}

function safeDataId(value) {
  return createHash("md5").update(`${value}:${Date.now()}:${randomUUID()}`).digest("hex").slice(0, 120);
}

function sanitizeArtifactStem(value) {
  const stem = path.basename(value, path.extname(value)) || "mineru_result";
  return sanitizeFilename(stem).replace(/\s+/gu, "_").slice(0, 120) || "mineru_result";
}

function sanitizeZipMemberName(value) {
  const parts = String(value || "")
    .replace(/\\/gu, "/")
    .split("/")
    .filter((part) => part && part !== "." && part !== "..");
  return sanitizeFilename(parts.join("__") || "artifact").slice(0, 180);
}

async function pdfParts(filePath, outputDir, maxPages = MAX_PAGES_PER_TASK) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext !== ".pdf") return [{ filePath, pageStart: 1, pageEnd: 1, pageOffset: 0 }];
  const bytes = await fs.readFile(filePath);
  const pdf = await PDFDocument.load(bytes, { ignoreEncryption: true });
  const pageCount = pdf.getPageCount();
  if (pageCount <= maxPages) return [{ filePath, pageStart: 1, pageEnd: pageCount, pageOffset: 0, pageCount }];

  await ensureDir(outputDir);
  const parts = [];
  for (let start = 0; start < pageCount; start += maxPages) {
    const end = Math.min(start + maxPages, pageCount);
    const out = await PDFDocument.create();
    const indexes = Array.from({ length: end - start }, (_, index) => start + index);
    const copied = await out.copyPages(pdf, indexes);
    for (const page of copied) out.addPage(page);
    const partPath = await uniquePath(
      path.join(outputDir, `${path.basename(filePath, ext)}_part${parts.length + 1}_${start + 1}-${end}.pdf`)
    );
    await fs.writeFile(partPath, await out.save());
    parts.push({ filePath: partPath, pageStart: start + 1, pageEnd: end, pageOffset: start, pageCount });
  }
  return parts;
}

async function prepareUploadParts(inputPaths, workDir, progress = () => {}) {
  const splitDir = path.join(workDir, "mineru_parts");
  const uploadParts = [];
  for (const originalPath of inputPaths) {
    const ext = path.extname(originalPath).toLowerCase();
    if (!MINERU_SUPPORTED_EXTENSIONS.has(ext)) continue;
    if (!(await pathExists(originalPath))) throw new MineruError(`Input file not found: ${originalPath}`);
    const parts = await pdfParts(originalPath, splitDir);
    if (ext === ".pdf") progress(`${path.basename(originalPath)}: PDF page count ${parts[0]?.pageCount ?? parts.length}`);
    const baseId = safeDataId(path.basename(originalPath));
    for (let index = 0; index < parts.length; index += 1) {
      const part = parts[index];
      const stat = await fs.stat(part.filePath);
      uploadParts.push({
        originalPath,
        uploadPath: part.filePath,
        uploadName: path.basename(part.filePath),
        dataId: `${baseId}_p${index + 1}`,
        partIndex: index + 1,
        partTotal: parts.length,
        pageStart: part.pageStart,
        pageEnd: part.pageEnd,
        pageOffset: part.pageOffset,
        sizeBytes: stat.size
      });
    }
  }
  return uploadParts;
}

function groupBatches(parts) {
  const batches = [];
  let current = [];
  let currentBytes = 0;
  for (const part of parts) {
    const countExceeded = current.length >= MAX_FILES_PER_BATCH;
    const bytesExceeded = current.length > 0 && currentBytes + part.sizeBytes > MAX_UPLOAD_BATCH_BYTES;
    if (countExceeded || bytesExceeded) {
      batches.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(part);
    currentBytes += part.sizeBytes;
  }
  if (current.length) batches.push(current);
  return batches;
}

async function createBatchUploadUrls(batch, options) {
  const data = await mineruJson(
    "POST",
    options.apiBase,
    options.token,
    "/file-urls/batch",
    {
      files: batch.map((part) => ({
        name: safeMineruFilename(part.uploadName),
        data_id: part.dataId,
        is_ocr: Boolean(options.isOcr)
      })),
      model_version: options.modelVersion,
      language: options.language,
      enable_formula: Boolean(options.enableFormula),
      enable_table: Boolean(options.enableTable)
    },
    options.timeout
  );
  const batchId = data?.data?.batch_id;
  const fileUrls = data?.data?.file_urls;
  if (!batchId || !Array.isArray(fileUrls) || fileUrls.length !== batch.length) {
    throw new MineruError(`MinerU returned invalid upload URL payload: ${JSON.stringify(data).slice(0, 800)}`);
  }
  return { batchId, fileUrls };
}

async function uploadFile(uploadUrl, filePath, timeout = 300) {
  const bytes = await fs.readFile(filePath);
  const response = await fetch(uploadUrl, {
    method: "PUT",
    body: bytes,
    signal: AbortSignal.timeout(timeout * 1000)
  });
  if (![200, 201, 204].includes(response.status)) {
    await assertOk(response, "MinerU file upload");
    throw new MineruError(`MinerU file upload failed: HTTP ${response.status}`);
  }
}

async function getBatchStatus(batchId, options) {
  return mineruJson("GET", options.apiBase, options.token, `/extract-results/batch/${encodeURIComponent(batchId)}`, null, options.timeout);
}

function zipEntries(buffer) {
  const entries = unzipSync(new Uint8Array(buffer));
  return Object.fromEntries(Object.entries(entries).map(([name, bytes]) => [name, Buffer.from(bytes)]));
}

async function downloadArtifacts(zipUrl, part, outputDir, timeout = 300) {
  const response = await fetch(zipUrl, { signal: AbortSignal.timeout(timeout * 1000) });
  await assertOk(response, "MinerU result download");
  const zipBuffer = Buffer.from(await response.arrayBuffer());
  const zipDir = path.join(outputDir, "mineru_zips");
  const jsonDir = path.join(outputDir, "mineru_json", sanitizeArtifactStem(part.uploadName));
  await ensureDir(zipDir);
  await ensureDir(jsonDir);

  const savedZip = await uniquePath(path.join(zipDir, `${sanitizeArtifactStem(part.uploadName)}.zip`));
  await fs.writeFile(savedZip, zipBuffer);

  const entries = zipEntries(zipBuffer);
  const mdNames = Object.keys(entries).filter((name) => /\.(md|markdown)$/iu.test(name));
  if (!mdNames.length) throw new MineruError("MinerU result zip contains no Markdown file");
  const preferred = mdNames.find((name) => path.basename(name).toLowerCase() === "full.md") || mdNames[0];
  const markdown = entries[preferred].toString("utf8").trim();

  const jsonPaths = [];
  let contentListPath = "";
  let middleJsonPath = "";
  for (const [name, bytes] of Object.entries(entries)) {
    if (!name.toLowerCase().endsWith(".json")) continue;
    const target = await uniquePath(path.join(jsonDir, sanitizeZipMemberName(name)));
    await ensureDir(path.dirname(target));
    await fs.writeFile(target, bytes);
    jsonPaths.push(target);
    const lower = name.toLowerCase();
    if (/_content_list\.json$/u.test(lower)) contentListPath = target;
    else if (!middleJsonPath && (/_middle\.json$/u.test(lower) || /middle.*\.json$/u.test(lower))) middleJsonPath = target;
  }
  if (!middleJsonPath) {
    middleJsonPath = jsonPaths.find((item) => !/_content_list\.json$/iu.test(path.basename(item))) || "";
  }
  return {
    markdown,
    markdownName: preferred,
    zipPath: savedZip,
    jsonPaths,
    contentListPath,
    middleJsonPath,
    pageOffset: part.pageOffset
  };
}

function inferDataIdByFilename(fileName, partByDataId) {
  if (!fileName) return "";
  for (const [dataId, part] of partByDataId.entries()) {
    if (fileName === part.uploadName || fileName === safeMineruFilename(part.uploadName)) return dataId;
  }
  return "";
}

async function pollBatchUntilDone(batchId, partByDataId, artifacts, failures, outputDir, options, progress = () => {}) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  const seenDone = new Set();
  while (Date.now() < deadline) {
    const status = await getBatchStatus(batchId, options);
    const extractResults = status?.data?.extract_result || [];
    if (!Array.isArray(extractResults) || !extractResults.length) {
      await sleep(POLL_INTERVAL_MS);
      continue;
    }
    for (const item of extractResults) {
      const dataId = String(item?.data_id || inferDataIdByFilename(item?.file_name, partByDataId));
      if (!dataId || seenDone.has(dataId)) continue;
      const state = String(item?.state || item?.status || "").toLowerCase();
      if (state === "done" || state === "success") {
        const zipUrl = item?.full_zip_url || item?.zip_url || item?.result_url;
        const part = partByDataId.get(dataId);
        if (!zipUrl) failures.set(dataId, "done but missing full_zip_url");
        else if (!part) failures.set(dataId, "done but upload part cannot be matched");
        else {
          progress(`MinerU: downloading result ${part.uploadName}`);
          artifacts.set(dataId, await downloadArtifacts(zipUrl, part, outputDir, Math.max(options.timeout, 300)));
        }
        seenDone.add(dataId);
      } else if (state === "failed" || state === "error") {
        failures.set(dataId, item?.err_msg || item?.message || "MinerU parse failed");
        seenDone.add(dataId);
      } else if (item?.extract_progress) {
        const page = item.extract_progress;
        progress(`MinerU: ${item.file_name || dataId} running ${page.extracted_pages ?? 0}/${page.total_pages ?? "?"} pages`);
      }
    }

    const expectedIds = new Set();
    for (const item of extractResults) {
      const dataId = String(item?.data_id || inferDataIdByFilename(item?.file_name, partByDataId));
      if (dataId && partByDataId.has(dataId)) expectedIds.add(dataId);
    }
    if (expectedIds.size && [...expectedIds].every((id) => artifacts.has(id) || failures.has(id))) return;
    await sleep(POLL_INTERVAL_MS);
  }
  throw new MineruError(`MinerU batch timeout: ${batchId}`);
}

async function readJsonMaybe(filePath) {
  if (!filePath) return null;
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch {
    return null;
  }
}

async function buildCombinedMineruJson(successParts, artifacts) {
  const contentList = [];
  const pdfInfo = [];
  const sourceParts = [];
  for (const part of successParts) {
    const artifact = artifacts.get(part.dataId);
    const rawContent = await readJsonMaybe(artifact.contentListPath);
    const contentItems = Array.isArray(rawContent) ? rawContent : Array.isArray(rawContent?.content_list) ? rawContent.content_list : [];
    for (const item of contentItems) {
      if (item && typeof item === "object") {
        const copied = JSON.parse(JSON.stringify(item));
        if (Number.isInteger(copied.page_idx)) copied.page_idx += part.pageOffset;
        contentList.push(copied);
      }
    }
    const middle = await readJsonMaybe(artifact.middleJsonPath);
    if (Array.isArray(middle?.pdf_info)) {
      for (const page of middle.pdf_info) {
        const copied = JSON.parse(JSON.stringify(page));
        if (Number.isInteger(copied.page_idx)) copied.page_idx += part.pageOffset;
        pdfInfo.push(copied);
      }
    }
    sourceParts.push({
      upload_name: part.uploadName,
      page_start: part.pageStart,
      page_end: part.pageEnd,
      page_offset: part.pageOffset,
      content_list_path: artifact.contentListPath,
      middle_json_path: artifact.middleJsonPath,
      zip_path: artifact.zipPath
    });
  }
  const combined = { content_list: contentList, source_parts: sourceParts };
  if (pdfInfo.length) combined.pdf_info = pdfInfo;
  return combined;
}

function fallbackMetadata(markdown, successParts) {
  return {
    source: "mineru",
    format: "rageval_mineru_metadata_node_v1",
    page_spans: successParts.length ? [{ page: 1, start: 0, end: markdown.length }] : [],
    semantic_spans: []
  };
}

export async function processWithMineru(inputPaths, outputDir, rawOptions = {}, progress = () => {}) {
  await ensureDir(outputDir);
  const options = {
    token: await readMineruToken(rawOptions.token),
    apiBase: rawOptions.apiBase || MINERU_API_BASE,
    modelVersion: rawOptions.modelVersion || "vlm",
    htmlModelVersion: rawOptions.htmlModelVersion || "MinerU-HTML",
    language: rawOptions.language || "ch",
    isOcr: rawOptions.isOcr !== false,
    enableFormula: rawOptions.enableFormula !== false,
    enableTable: rawOptions.enableTable !== false,
    timeout: Number.parseInt(String(rawOptions.timeout || 120), 10) || 120
  };

  const parts = await prepareUploadParts(inputPaths, outputDir, progress);
  if (!parts.length) return [];
  const partByDataId = new Map(parts.map((part) => [part.dataId, part]));
  const artifacts = new Map();
  const failures = new Map();
  const groups = {
    document: parts.filter((part) => ![".html", ".htm"].includes(path.extname(part.uploadName).toLowerCase())),
    html: parts.filter((part) => [".html", ".htm"].includes(path.extname(part.uploadName).toLowerCase()))
  };

  for (const [groupName, groupParts] of Object.entries(groups)) {
    if (!groupParts.length) continue;
    const modelVersion = groupName === "html" ? options.htmlModelVersion : options.modelVersion;
    for (const [batchIndex, batch] of groupBatches(groupParts).entries()) {
      progress(`MinerU: requesting upload URLs for ${groupName} batch ${batchIndex + 1}, files=${batch.length}`);
      const { batchId, fileUrls } = await createBatchUploadUrls(batch, { ...options, modelVersion });
      for (let index = 0; index < batch.length; index += 1) {
        progress(`MinerU: uploading ${batch[index].uploadName}`);
        await uploadFile(fileUrls[index], batch[index].uploadPath, Math.max(options.timeout, 300));
      }
      progress(`MinerU: polling batch ${batchId}`);
      await pollBatchUntilDone(batchId, partByDataId, artifacts, failures, outputDir, options, progress);
    }
  }

  const partsByOriginal = new Map();
  for (const part of parts) {
    if (!partsByOriginal.has(part.originalPath)) partsByOriginal.set(part.originalPath, []);
    partsByOriginal.get(part.originalPath).push(part);
  }

  const results = [];
  for (const [originalPath, originalParts] of partsByOriginal.entries()) {
    originalParts.sort((a, b) => a.partIndex - b.partIndex);
    const successParts = originalParts.filter((part) => artifacts.has(part.dataId));
    const failedParts = originalParts.filter((part) => failures.has(part.dataId));
    if (!successParts.length) {
      results.push({
        original_path: originalPath,
        success: false,
        part_count: originalParts.length,
        error: originalParts.map((part) => failures.get(part.dataId) || "not completed").join("; "),
        parts: originalParts
      });
      continue;
    }

    const merged = successParts
      .map((part) => `<!-- source: ${path.basename(originalPath)}; pages: ${part.pageStart}-${part.pageEnd}; part: ${part.partIndex}/${part.partTotal} -->\n\n${artifacts.get(part.dataId).markdown}`)
      .join("\n\n")
      .trim();
    const markdownPath = await uniquePath(path.join(outputDir, `${path.basename(originalPath, path.extname(originalPath))}.mineru.md`));
    await fs.writeFile(markdownPath, `${merged}\n`, "utf8");
    const metadataPath = `${markdownPath}.meta.json`;
    await writeJson(metadataPath, fallbackMetadata(merged, successParts));
    const combinedJsonPath = `${markdownPath}.mineru.json`;
    await writeJson(combinedJsonPath, await buildCombinedMineruJson(successParts, artifacts));

    const zipPaths = successParts.map((part) => artifacts.get(part.dataId).zipPath);
    const jsonPaths = successParts.flatMap((part) => artifacts.get(part.dataId).jsonPaths);
    results.push({
      original_path: originalPath,
      success: true,
      markdown_path: markdownPath,
      metadata_path: metadataPath,
      zip_paths: zipPaths,
      json_paths: jsonPaths,
      content_list_paths: successParts.map((part) => artifacts.get(part.dataId).contentListPath).filter(Boolean),
      middle_json_paths: successParts.map((part) => artifacts.get(part.dataId).middleJsonPath).filter(Boolean),
      combined_json_path: combinedJsonPath,
      text_chars: merged.length,
      part_count: originalParts.length,
      error: failedParts.length ? `partial MinerU failure: ${failedParts.map((part) => failures.get(part.dataId)).join("; ")}` : "",
      parts: originalParts
    });
  }
  return results;
}

export function mineruSupportedPaths(paths) {
  return paths.filter((filePath) => MINERU_SUPPORTED_EXTENSIONS.has(path.extname(filePath).toLowerCase()));
}
