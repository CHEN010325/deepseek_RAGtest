import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { ASSETS_DIR, DATA_DIR } from "../config.js";
import { isoTimestamp, pathExists, readJson, readText, sanitizeDatasetName, writeJson } from "../utils/files.js";

const EMBEDDING_SAFE_TOKEN_LIMIT = 24000;

function estimateDenseTokens(text) {
  let total = 0;
  for (const char of String(text || "")) {
    if (/\s/u.test(char)) continue;
    if (/[\u4e00-\u9fff]/u.test(char)) total += 1.5;
    else if (/[A-Za-z0-9]/u.test(char)) total += 0.45;
    else if (char.codePointAt(0) < 128) total += 0.8;
    else total += 1;
  }
  return Math.ceil(total);
}

export function estimateTokens(text) {
  const value = String(text || "");
  if (!value) return 0;
  const chineseChars = (value.match(/[\u4e00-\u9fff]/gu) || []).length;
  const latinWords = (value.match(/[A-Za-z]+(?:'[A-Za-z]+)?/gu) || []).length;
  const numberGroups = (value.match(/\d+(?:[.,:/_-]\d+)*/gu) || []).length;
  const wordEstimate = Math.ceil(chineseChars * 1.5 + latinWords * 1.3 + numberGroups);
  return Math.max(wordEstimate, estimateDenseTokens(value));
}

function splitByParagraphs(text) {
  return String(text || "").split(/\n\s*\n/gu).map((part) => part.trim()).filter(Boolean);
}

function splitByLines(text) {
  return String(text || "").split(/\n+/gu).map((part) => part.trim()).filter(Boolean);
}

function splitBySentences(text) {
  return (String(text || "").match(/[^。！？?!\n]+[。！？?!\n]*/gu) || []).map((part) => part.trim()).filter(Boolean);
}

function splitByWords(text) {
  return (String(text || "").match(/\S+\s*/gu) || []).map((part) => part.trim()).filter(Boolean);
}

function isGfmTableSeparator(line) {
  const trimmed = String(line || "").trim();
  if (!trimmed.includes("|")) return false;
  const cells = trimmed.replace(/^\|/u, "").replace(/\|$/u, "").split("|").map((cell) => cell.trim());
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/u.test(cell || ""));
}

function isCompleteHtmlTableBlock(text) {
  return /^<table\b[\s\S]*<\/table>\s*$/iu.test(String(text || "").trim());
}

function isCompleteGfmTableBlock(text) {
  const lines = String(text || "").trim().split(/\r?\n/u).map((line) => line.trim()).filter(Boolean);
  if (lines.length < 2) return false;
  const separatorIndex = lines.findIndex((line) => isGfmTableSeparator(line));
  return separatorIndex >= 1 && lines.every((line) => line.includes("|"));
}

function isAtomicTableBlock(text) {
  const normalized = String(text || "").trim();
  return isCompleteHtmlTableBlock(normalized) || isCompleteGfmTableBlock(normalized);
}

function hardSplitByCharacterWindow(text, chunkTokenNum) {
  const normalized = String(text || "").trim();
  if (!normalized) return [];
  const safeCharWindow = Math.max(200, Math.floor(chunkTokenNum * 2));
  const parts = [];
  let start = 0;
  while (start < normalized.length) {
    let end = Math.min(normalized.length, start + safeCharWindow);
    let candidate = normalized.slice(start, end).trim();
    while (candidate && estimateTokens(candidate) > chunkTokenNum && end > start + 1) {
      end -= Math.max(1, Math.floor((end - start) / 4));
      candidate = normalized.slice(start, end).trim();
    }
    if (!candidate) {
      end = Math.min(normalized.length, start + 1);
      candidate = normalized.slice(start, end).trim();
    }
    if (candidate) parts.push(candidate);
    start = end;
  }
  return parts;
}

function packSegmentsByTokenBudget(segments, chunkTokenNum, joiner, nextStrategyIndex, strategies, preserveAtomicTables = true) {
  const packed = [];
  let current = "";
  const flush = () => {
    const normalized = current.trim();
    if (normalized) packed.push(normalized);
    current = "";
  };

  for (const segment of segments) {
    const normalizedSegment = String(segment || "").trim();
    if (!normalizedSegment) continue;
    if (estimateTokens(normalizedSegment) > chunkTokenNum) {
      flush();
      packed.push(
        ...splitOversizedMarkdownChunk(
          normalizedSegment,
          chunkTokenNum,
          nextStrategyIndex,
          strategies,
          preserveAtomicTables
        )
      );
      continue;
    }
    const candidate = current ? `${current}${joiner}${normalizedSegment}` : normalizedSegment;
    if (current && estimateTokens(candidate) > chunkTokenNum) {
      flush();
      current = normalizedSegment;
    } else {
      current = candidate;
    }
  }
  flush();
  return packed;
}

function splitOversizedMarkdownChunk(
  text,
  chunkTokenNum,
  strategyIndex = 0,
  strategies = null,
  preserveAtomicTables = true
) {
  const normalized = String(text || "").trim();
  if (!normalized) return [];
  if (estimateTokens(normalized) <= chunkTokenNum) return [normalized];
  if (preserveAtomicTables && isAtomicTableBlock(normalized)) return [normalized];
  const activeStrategies = strategies || [
    [splitByParagraphs, "\n\n"],
    [splitByLines, "\n"],
    [splitBySentences, "\n"],
    [splitByWords, " "],
    [(value) => Array.from(value), ""]
  ];
  if (strategyIndex >= activeStrategies.length) return hardSplitByCharacterWindow(normalized, chunkTokenNum);
  const [splitter, joiner] = activeStrategies[strategyIndex];
  const segments = splitter(normalized);
  if (segments.length <= 1) {
    return splitOversizedMarkdownChunk(
      normalized,
      chunkTokenNum,
      strategyIndex + 1,
      activeStrategies,
      preserveAtomicTables
    );
  }
  return packSegmentsByTokenBudget(
    segments,
    chunkTokenNum,
    joiner,
    strategyIndex + 1,
    activeStrategies,
    preserveAtomicTables
  );
}

function enforceChunkTokenLimit(chunks, chunkTokenNum) {
  const out = [];
  for (const chunk of chunks) out.push(...splitOversizedMarkdownChunk(chunk, chunkTokenNum));
  return out.map((chunk) => chunk.trim()).filter(Boolean);
}

function splitMarkdownBasic(text, chunkTokenNum = 275, minChunkTokens = 50) {
  if (isAtomicTableBlock(text)) return [String(text || "").trim()];
  const chunks = [];
  let current = [];
  let currentTokens = 0;
  for (const paragraph of String(text || "").split(/\n\s*\n/gu).filter(Boolean)) {
    const paraTokens = estimateTokens(paragraph);
    if (paraTokens > chunkTokenNum) {
      if (current.length) {
        chunks.push(current.join("\n\n"));
        current = [];
        currentTokens = 0;
      }
      let sentenceChunk = [];
      let sentenceTokens = 0;
      for (const sentence of paragraph.split(/(?<=[。！？?!])/u)) {
        if (!sentence) continue;
        const sentTokens = estimateTokens(sentence);
        if (sentenceChunk.length && sentenceTokens + sentTokens > chunkTokenNum) {
          chunks.push(sentenceChunk.join(""));
          sentenceChunk = [sentence];
          sentenceTokens = sentTokens;
        } else {
          sentenceChunk.push(sentence);
          sentenceTokens += sentTokens;
        }
      }
      if (sentenceChunk.length) chunks.push(sentenceChunk.join(""));
      continue;
    }
    if (current.length && currentTokens + paraTokens > chunkTokenNum) {
      chunks.push(current.join("\n\n"));
      current = [paragraph];
      currentTokens = paraTokens;
    } else {
      current.push(paragraph);
      currentTokens += paraTokens;
    }
  }
  if (current.length && (currentTokens >= minChunkTokens || !chunks.length)) chunks.push(current.join("\n\n"));
  const filtered = enforceChunkTokenLimit(chunks, chunkTokenNum);
  if (!filtered.length && String(text || "").trim()) return splitOversizedMarkdownChunk(String(text).trim(), chunkTokenNum);
  return filtered;
}

function splitMarkdownByTitle(text, chunkTokenNum = 275, minChunkTokens = 50) {
  const matches = [...String(text || "").matchAll(/^#{1,6}\s+.+$/gmu)];
  if (!matches.length) return splitMarkdownBasic(text, chunkTokenNum, minChunkTokens);
  const chunks = [];
  const firstTitleStart = matches[0].index || 0;
  if (firstTitleStart > 0) {
    const prefix = String(text).slice(0, firstTitleStart).trim();
    if (prefix) {
      const prefixTokens = estimateTokens(prefix);
      if (prefixTokens > chunkTokenNum) chunks.push(...splitMarkdownBasic(prefix, chunkTokenNum, minChunkTokens));
      else if (prefixTokens >= minChunkTokens || !chunks.length) chunks.push(prefix);
    }
  }
  for (let index = 0; index < matches.length; index += 1) {
    const sectionStart = matches[index].index || 0;
    const sectionEnd = index + 1 < matches.length ? matches[index + 1].index || String(text).length : String(text).length;
    const section = String(text).slice(sectionStart, sectionEnd).trim();
    const sectionTokens = estimateTokens(section);
    if (sectionTokens > chunkTokenNum) {
      let paraChunk = [];
      let paraTokens = 0;
      for (const paragraph of section.split(/\n\s*\n/gu)) {
        const paragraphTokens = estimateTokens(paragraph);
        if (paraChunk.length && paraTokens + paragraphTokens > chunkTokenNum) {
          chunks.push(paraChunk.join("\n\n"));
          paraChunk = [paragraph];
          paraTokens = paragraphTokens;
        } else {
          paraChunk.push(paragraph);
          paraTokens += paragraphTokens;
        }
      }
      if (paraChunk.length) chunks.push(paraChunk.join("\n\n"));
    } else if (sectionTokens >= minChunkTokens || !chunks.length) {
      chunks.push(section);
    }
  }
  const filtered = enforceChunkTokenLimit(chunks, chunkTokenNum);
  if (!filtered.length && String(text || "").trim()) return splitOversizedMarkdownChunk(String(text).trim(), chunkTokenNum);
  return filtered;
}

function normalizeForSearch(value) {
  let text = "";
  const indexMap = [];
  let prevWhitespace = false;
  const source = String(value || "");
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (/\s/u.test(char)) {
      if (!prevWhitespace) {
        text += " ";
        indexMap.push(index);
        prevWhitespace = true;
      }
      continue;
    }
    text += char;
    indexMap.push(index);
    prevWhitespace = false;
  }
  return [text, indexMap];
}

function locateMarkdownChunkRanges(text, chunks) {
  const exact = [];
  let cursor = 0;
  let exactSuccess = true;
  for (const content of chunks) {
    const exactIndex = String(text).indexOf(content, cursor);
    if (exactIndex < 0) {
      exactSuccess = false;
      break;
    }
    exact.push({ content, start: exactIndex, end: exactIndex + content.length });
    cursor = exactIndex + content.length;
  }
  if (exactSuccess) return exact;

  const [normalizedSource, indexMap] = normalizeForSearch(text);
  const results = [];
  let normalizedCursor = 0;
  for (const content of chunks) {
    const [normalizedRaw] = normalizeForSearch(content);
    const normalizedChunk = normalizedRaw.trim();
    if (!normalizedChunk) {
      results.push({ content, start: -1, end: -1 });
      continue;
    }
    const normalizedIndex = normalizedSource.indexOf(normalizedChunk, normalizedCursor);
    if (normalizedIndex < 0) {
      results.push({ content, start: -1, end: -1 });
      continue;
    }
    const start = normalizedIndex < indexMap.length ? indexMap[normalizedIndex] : -1;
    const endMapIndex = normalizedIndex + normalizedChunk.length - 1;
    const end = endMapIndex >= 0 && endMapIndex < indexMap.length ? indexMap[endMapIndex] + 1 : start + content.length;
    results.push({ content, start, end });
    normalizedCursor = normalizedIndex + normalizedChunk.length;
  }
  return results;
}

export function splitChunksWithMetadata(text, fileType = "md") {
  if (["md", "markdown"].includes(String(fileType || "").toLowerCase())) {
    const chunkTokenNum = 1000;
    const textTokens = estimateTokens(text);
    const defaultMinChunkTokens = Math.max(50, 275 / 5);
    const minChunkTokens = textTokens < defaultMinChunkTokens ? Math.min(10, Math.max(1, textTokens / 2)) : defaultMinChunkTokens;
    const chunks = splitMarkdownByTitle(text, Math.trunc(chunkTokenNum), Math.trunc(minChunkTokens));
    return locateMarkdownChunkRanges(text, chunks);
  }
  return locateMarkdownChunkRanges(text, splitMarkdownBasic(text, 275, 55));
}

function trimUploadChunk(rawText, start, end) {
  const text = String(rawText || "");
  let left = Math.max(0, Math.min(text.length, Math.trunc(Number(start) || 0)));
  let right = Math.max(left, Math.min(text.length, Math.trunc(Number(end) || left)));
  while (left < right && /\s/u.test(text[left])) left += 1;
  while (right > left && /\s/u.test(text[right - 1])) right -= 1;
  if (right <= left) return null;
  return { content: text.slice(left, right), start: left, end: right };
}

function isHeadingOnlyChunk(content) {
  const lines = String(content || "").trim().split(/\r?\n/u).map((line) => line.trim()).filter(Boolean);
  return lines.length === 1 && /^#{1,6}\s+\S/u.test(lines[0]);
}

function mergeHeadingOnlyChunks(chunks) {
  const merged = [];
  let index = 0;
  while (index < chunks.length) {
    const current = chunks[index];
    const nextChunk = index + 1 < chunks.length ? chunks[index + 1] : null;
    if (nextChunk && isHeadingOnlyChunk(String(current.content || ""))) {
      merged.push({
        content: `${String(current.content).trim()}\n\n${String(nextChunk.content).trim()}`,
        start: current.start,
        end: nextChunk.end
      });
      index += 2;
      continue;
    }
    merged.push(current);
    index += 1;
  }
  return merged;
}

function splitTextRangeByOriginalMarkdown(rawText, start, end, fileType) {
  const trimmed = trimUploadChunk(rawText, start, end);
  if (!trimmed) return [];
  const chunks = [];
  for (const chunk of splitChunksWithMetadata(trimmed.content, fileType)) {
    chunks.push({
      content: chunk.content,
      start: trimmed.start + Math.max(0, Math.trunc(Number(chunk.start) || 0)),
      end: trimmed.start + Math.max(0, Math.trunc(Number(chunk.end) || 0))
    });
  }
  return String(fileType || "").toLowerCase() === "md" ? mergeHeadingOnlyChunks(chunks) : chunks;
}

function expandImageSpan(rawText, span) {
  const text = String(rawText || "");
  let start = Math.max(0, Math.min(text.length, Math.trunc(Number(span?.start) || 0)));
  const end = Math.max(start, Math.min(text.length, Math.trunc(Number(span?.end) || start)));
  const lookBehindStart = Math.max(0, start - 2000);
  const lookBehind = text.slice(lookBehindStart, start);
  const match = lookBehind.match(/(?:^|\r?\n)[ \t]*(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\r?\n[ \t]*)*$/iu);
  if (match?.index !== undefined) start = lookBehindStart + match.index;
  return [start, end];
}

function trimTableSpanBeforeTrailingImage(rawText, start, end) {
  const textSlice = String(rawText || "").slice(start, end);
  const match = textSlice.match(/(?:\r?\n[ \t]*)+(?:!\[[^\]\r\n]*\]\([^)]+\)|<img\b[\s\S]*?>)[ \t]*(?:\r?\n[ \t]*)*$/iu);
  if (!match?.index) return [start, end];
  if (!/<table\b/iu.test(textSlice.slice(0, match.index))) return [start, end];
  return [start, start + match.index];
}

function normalizeAtomicSemanticSpan(rawText, span) {
  const spanType = String(span?.type || "").toLowerCase();
  if (!["table", "image"].includes(spanType)) return null;
  const text = String(rawText || "");
  const initialStart = Math.max(0, Math.min(text.length, Math.trunc(Number(span?.start) || 0)));
  const initialEnd = Math.max(initialStart, Math.min(text.length, Math.trunc(Number(span?.end) || initialStart)));
  const expanded = spanType === "table" ? trimTableSpanBeforeTrailingImage(text, initialStart, initialEnd) : expandImageSpan(text, span);
  const chunk = trimUploadChunk(text, expanded[0], expanded[1]);
  if (chunk) chunk.atomicType = spanType;
  return chunk;
}

function clampSpanToMarkdownSection(rawText, span) {
  const text = String(rawText || "");
  let sectionStart = 0;
  let sectionEnd = text.length;
  for (const match of text.matchAll(/^#{1,6}\s+\S.*$/gmu)) {
    const headingStart = match.index || 0;
    if (headingStart <= span.start) {
      sectionStart = headingStart;
      continue;
    }
    sectionEnd = headingStart;
    break;
  }
  const trimmed = trimUploadChunk(text, Math.max(sectionStart, span.start), Math.min(sectionEnd, span.end));
  if (trimmed) trimmed.atomicType = span.atomicType;
  return trimmed;
}

function getOrderedAtomicSemanticSpans(rawText, semanticSpans) {
  const spans = [];
  for (const span of Array.isArray(semanticSpans) ? semanticSpans : []) {
    try {
      if (Number(span?.end || 0) <= Number(span?.start || 0)) continue;
      const normalized = normalizeAtomicSemanticSpan(rawText, span);
      if (!normalized) continue;
      const clamped = clampSpanToMarkdownSection(rawText, normalized);
      if (clamped) spans.push(clamped);
    } catch {
      // Ignore malformed MinerU spans.
    }
  }
  return spans.sort((a, b) => Number(a.start) - Number(b.start));
}

function mergeMarkdownChunksAroundAtomicSpans(rawText, markdownChunks, atomicSpans) {
  const chunks = [...markdownChunks].sort((a, b) => Number(a.start) - Number(b.start));
  for (const span of atomicSpans) {
    const first = chunks.findIndex((chunk) => chunk.end > span.start && chunk.start < span.end);
    if (first < 0) {
      const insertAt = chunks.findIndex((chunk) => chunk.start > span.start);
      chunks.splice(insertAt < 0 ? chunks.length : insertAt, 0, span);
      continue;
    }
    let last = first;
    while (last + 1 < chunks.length && chunks[last + 1].start < span.end) last += 1;
    const merged = trimUploadChunk(
      rawText,
      Math.min(chunks[first].start, span.start),
      Math.max(chunks[last].end, span.end)
    );
    if (merged) chunks.splice(first, last - first + 1, merged);
  }
  return chunks;
}

function splitBySemanticSpans(rawText, semanticSpans, fileType) {
  const markdownChunks = splitTextRangeByOriginalMarkdown(rawText, 0, String(rawText || "").length, fileType);
  const atomicSpans = getOrderedAtomicSemanticSpans(rawText, semanticSpans);
  if (!atomicSpans.length) return markdownChunks;
  return mergeMarkdownChunksAroundAtomicSpans(rawText, markdownChunks, atomicSpans) || markdownChunks;
}

function normalizeSpans(spans, pageRequired) {
  if (!Array.isArray(spans)) return [];
  const out = [];
  for (const span of spans) {
    if (!span || typeof span !== "object") continue;
    const start = Number.parseInt(String(span.start), 10);
    const end = Number.parseInt(String(span.end), 10);
    if (!Number.isInteger(start) || !Number.isInteger(end) || end <= start) continue;
    const normalized = { start, end };
    if (span.page !== undefined && span.page !== null) {
      const page = Number.parseInt(String(span.page), 10);
      if (Number.isInteger(page)) normalized.page = page;
    } else if (pageRequired) {
      continue;
    }
    const spanType = String(span.type || "").trim();
    if (spanType) normalized.type = spanType;
    const title = String(span.title || "").trim();
    if (title) normalized.title = title;
    out.push(normalized);
  }
  return out;
}

async function loadMineruMetadata(markdownPath) {
  const candidates = [
    `${markdownPath}.meta.json`,
    path.join(path.dirname(markdownPath), `${path.basename(markdownPath)}.meta.json`),
    path.join(path.dirname(markdownPath), `${path.basename(markdownPath, path.extname(markdownPath))}.meta.json`)
  ];
  for (const candidate of candidates) {
    const payload = await readJson(candidate, null);
    if (!payload || typeof payload !== "object") continue;
    const pageSpans = payload.page_spans || payload.pageSpans || [];
    const semanticSpans = payload.semantic_spans || payload.semanticSpans || [];
    return {
      pageSpans: normalizeSpans(pageSpans, true),
      semanticSpans: normalizeSpans(semanticSpans, false)
    };
  }
  return { pageSpans: [], semanticSpans: [] };
}

function getChunkPageValue(pageSpans, start, end) {
  if (!pageSpans?.length || start < 0 || end <= start) return "";
  const pages = new Set();
  for (const span of pageSpans) {
    if (Number(span.end) > start && Number(span.start) < end && span.page !== undefined) pages.add(Number(span.page));
  }
  const ordered = [...pages].filter(Number.isFinite).sort((a, b) => a - b);
  if (!ordered.length) return "";
  return ordered[0] === ordered[ordered.length - 1] ? String(ordered[0]) : `${ordered[0]}-${ordered[ordered.length - 1]}`;
}

function splitOversizedTableForEmbedding(text, chunkTokenNum) {
  const normalized = String(text || "").trim();
  if (!normalized) return [];
  if (!isAtomicTableBlock(normalized) || estimateTokens(normalized) <= chunkTokenNum) return [normalized];
  return splitOversizedMarkdownChunk(normalized, chunkTokenNum, 0, null, false);
}

export function buildMineruMarkdownChunkDocs(markdownText, docIndexBase, knowledgeLabel, pageSpans = [], semanticSpans = []) {
  const chunkMetas = semanticSpans.length
    ? splitBySemanticSpans(markdownText, semanticSpans, "md")
    : splitTextRangeByOriginalMarkdown(markdownText, 0, String(markdownText || "").length, "md");
  const docs = [];
  let idx = 1;
  for (const chunk of chunkMetas) {
    const original = String(chunk?.content || "");
    if (!original.trim()) continue;
    const stableId = crypto.createHash("sha1").update(`${knowledgeLabel}::${docIndexBase}::${idx}`).digest("hex");
    const page = getChunkPageValue(pageSpans, Number(chunk.start), Number(chunk.end));
    const embeddingParts = splitOversizedTableForEmbedding(original, EMBEDDING_SAFE_TOKEN_LIMIT);
    const isOversizedTable = embeddingParts.length > 1;
    const sourceDoc = {
      id: stableId,
      doc_index2: idx,
      title: docIndexBase,
      content: original
    };
    if (isOversizedTable) {
      sourceDoc.embeddingContent = embeddingParts[0] || original;
      sourceDoc.fragmentRole = "display";
      sourceDoc.sourceStart = Number(chunk.start);
      sourceDoc.sourceEnd = Number(chunk.end);
    }
    if (page) sourceDoc.page = page;
    docs.push(sourceDoc);
    if (isOversizedTable) {
      for (let partIndex = 2; partIndex <= embeddingParts.length; partIndex += 1) {
        const part = embeddingParts[partIndex - 1];
        const partDoc = {
          id: crypto.createHash("sha1").update(`${knowledgeLabel}::${docIndexBase}::${idx}::embedding::${partIndex}`).digest("hex"),
          doc_index2: idx + partIndex / 10000,
          title: docIndexBase,
          content: part,
          fragmentRole: "embedding_part",
          sourceDocIndex2: idx,
          sourceStart: Number(chunk.start),
          sourceEnd: Number(chunk.end),
          tablePart: partIndex,
          tablePartTotal: embeddingParts.length
        };
        if (page) partDoc.page = page;
        docs.push(partDoc);
      }
    }
    idx += 1;
  }
  return docs;
}

export async function buildChunkDocsFromMarkdownPath(markdownPath, knowledgeLabel) {
  const markdownText = await readText(markdownPath);
  const { pageSpans, semanticSpans } = await loadMineruMetadata(markdownPath);
  return buildMineruMarkdownChunkDocs(markdownText, path.basename(markdownPath), knowledgeLabel, pageSpans, semanticSpans);
}

export async function writeChunkSidecar(markdownPath, datasetName) {
  if (!(await pathExists(markdownPath)) || !(await pathExists(`${markdownPath}.mineru.json`))) return "";
  const docs = await buildChunkDocsFromMarkdownPath(markdownPath, `dryrun_${sanitizeDatasetName(datasetName)}`);
  const chunks = docs
    .filter((item) => String(item.content || "").trim())
    .map((item) => ({
      doc_index2: item.doc_index2,
      title: item.title,
      content: item.content,
      page: item.page || "",
      fragmentRole: item.fragmentRole || "",
      sourceStart: item.sourceStart ?? null,
      sourceEnd: item.sourceEnd ?? null,
      sourceDocIndex2: item.sourceDocIndex2 ?? null,
      tablePart: item.tablePart ?? null,
      tablePartTotal: item.tablePartTotal ?? null
    }));
  if (!chunks.length) return "";
  const sidecarPath = `${markdownPath}.deepseekmine_chunks.json`;
  await writeJson(sidecarPath, {
    schema: "deepseekmine-mineru-chunks-v1",
    source: "rageval-local-deeplocals-compatible-node",
    chunks
  });
  return sidecarPath;
}

export async function writeDeepLocalsChunkSidecars(markdownPaths, datasetName, progress = () => {}) {
  const sidecars = [];
  for (const markdownPath of markdownPaths || []) {
    try {
      const sidecar = await writeChunkSidecar(markdownPath, datasetName);
      if (sidecar) {
        sidecars.push(sidecar);
        progress(`生成 DeepLocals 分块 sidecar：${path.basename(markdownPath)} -> ${path.basename(sidecar)}`);
      }
    } catch (error) {
      progress(`DeepLocals 分块 sidecar 生成失败：${path.basename(markdownPath)} - ${error.message}`);
    }
  }
  return sidecars;
}

export async function loadAssetManifest(datasetName) {
  const manifestPath = path.join(ASSETS_DIR, sanitizeDatasetName(datasetName), "manifest.json");
  return readJson(manifestPath, null);
}

export async function assetMarkdownPaths(datasetName) {
  const manifest = await loadAssetManifest(datasetName);
  if (!manifest || typeof manifest !== "object") return [];
  const paths = [];
  for (const raw of manifest.mineru_markdown_files || []) {
    const markdownPath = path.resolve(String(raw));
    if ((await pathExists(markdownPath)) && (await pathExists(`${markdownPath}.mineru.json`))) paths.push(markdownPath);
  }
  return paths;
}

async function createNativeKnowledgeBase(apiBase, title, timeout = 60) {
  const response = await fetch(`${apiBase.replace(/\/+$/u, "")}/api/kb`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      date: isoTimestamp(),
      creator: "RAGEval Forge",
      icon: "RF",
      bgColor: "#0f766e"
    }),
    signal: AbortSignal.timeout(timeout * 1000)
  });
  const text = await response.text();
  if (response.status >= 400) throw new Error(`DeepLocals create KB failed: HTTP ${response.status} ${text.slice(0, 1000)}`);
  const payload = JSON.parse(text || "{}");
  const created = payload.created;
  if (!created || typeof created !== "object" || created.id === undefined || created.id === null) {
    throw new Error(`DeepLocals create KB failed: ${JSON.stringify(payload).slice(0, 1000)}`);
  }
  return created;
}

export async function uploadMarkdownAssetsViaNativeUpload(apiBase, knowledgeTitle, markdownPaths, timeout = 900, progress = () => {}) {
  const uniqueTitle = `${knowledgeTitle}_${Math.floor(Date.now() / 1000)}`;
  const created = await createNativeKnowledgeBase(apiBase, uniqueTitle, Math.min(timeout, 60));
  const kbId = String(created.id);
  await new Promise((resolve) => setTimeout(resolve, 1000));

  const form = new FormData();
  form.append("knowledgeLabel", kbId);
  form.append("waitForProcessing", "true");
  form.append("fastParsing", "true");
  for (const markdownPath of markdownPaths) {
    form.append("relativePath", path.basename(markdownPath));
    form.append("localPath", markdownPath);
    const bytes = await fs.readFile(markdownPath);
    form.append("file", new Blob([bytes], { type: "text/markdown" }), path.basename(markdownPath));
  }
  progress(`上传 MinerU Markdown 到 DeepLocals 原生接口：knowledgeLabel=${kbId}`);
  const response = await fetch(`${apiBase.replace(/\/+$/u, "")}/api/files/upload`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(timeout * 1000)
  });
  const text = await response.text();
  if (response.status >= 400) throw new Error(`DeepLocals native markdown upload failed: HTTP ${response.status} ${text.slice(0, 1000)}`);
  const payload = JSON.parse(text || "{}");
  if (payload.code !== 0) throw new Error(`DeepLocals native markdown upload failed: ${JSON.stringify(payload).slice(0, 1000)}`);
  let docIds = Array.isArray(payload.files) ? payload.files.map((item) => String(item).trim()).filter(Boolean) : [];
  if (!docIds.length) docIds = markdownPaths.map((item) => path.basename(item));
  return {
    knowledgeLabel: kbId,
    docIds,
    uploadPayload: {
      code: 0,
      method: "native_files_upload_with_mineru_sidecar",
      requestedKnowledgeLabel: knowledgeTitle,
      importedKnowledgeLabel: kbId,
      importedKnowledgeTitle: uniqueTitle,
      createdKnowledgeBase: created,
      files: docIds,
      uploadResponse: payload
    }
  };
}

export function sidecarPathForMarkdown(markdownPath) {
  return `${markdownPath}.deepseekmine_chunks.json`;
}

export async function sidecarSummaryForDataset(datasetName) {
  const markdownPaths = await assetMarkdownPaths(datasetName);
  const sidecars = [];
  for (const markdownPath of markdownPaths) {
    const sidecarPath = sidecarPathForMarkdown(markdownPath);
    if (await pathExists(sidecarPath)) sidecars.push(sidecarPath);
  }
  return {
    dataset_name: sanitizeDatasetName(datasetName),
    markdown_count: markdownPaths.length,
    sidecar_count: sidecars.length,
    markdown_paths: markdownPaths,
    sidecar_paths: sidecars,
    assets_dir: path.join(DATA_DIR, "assets", sanitizeDatasetName(datasetName))
  };
}
