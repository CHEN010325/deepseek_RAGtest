import fs from "node:fs/promises";
import path from "node:path";
import { DATA_DIR, TEMP_DOCS_DIR } from "../config.js";
import { ensureDir, pathExists, readJson, readText } from "../utils/files.js";
import { normalizeText } from "./scoring.js";

export async function loadJsonl(filePath, limit = 0) {
  const rows = [];
  const content = await readText(filePath);
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    rows.push(JSON.parse(line));
    if (limit && rows.length >= limit) break;
  }
  return rows;
}

export function parseIds(idsText = "") {
  return String(idsText)
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number.parseInt(part, 10));
}

export function filterRowsByIds(rows, ids = []) {
  if (!ids.length) return rows;
  const wanted = new Set(ids);
  const found = new Set(
    rows
      .map((row) => Number.parseInt(String(row.id ?? ""), 10))
      .filter((id) => Number.isInteger(id))
  );
  const missing = [...wanted].filter((id) => !found.has(id)).sort((a, b) => a - b);
  if (missing.length) {
    throw new Error(`Unknown QA ids: ${missing.join(", ")}`);
  }
  return rows.filter((row) => wanted.has(Number.parseInt(String(row.id ?? ""), 10)));
}

export function flatten(items) {
  if (Array.isArray(items)) return items.flatMap((item) => flatten(item));
  return [items];
}

export function answersFor(row) {
  return flatten(row.answer ?? [])
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
}

export function evidenceQuotesFor(row) {
  const quotes = [];
  if (Array.isArray(row.evidence)) {
    for (const item of row.evidence) {
      if (item && typeof item === "object" && String(item.quote ?? "").trim()) {
        quotes.push(String(item.quote).trim());
      } else if (typeof item === "string" && item.trim()) {
        quotes.push(item.trim());
      }
    }
  }
  if (!quotes.length) {
    quotes.push(
      ...flatten(row.positive ?? [])
        .map((item) => String(item ?? "").trim())
        .filter(Boolean)
    );
  }
  return quotes;
}

export async function resolveDatasetPath(datasetName, datasetPath = "") {
  const resolved = datasetPath ? path.resolve(datasetPath) : path.join(DATA_DIR, `${datasetName}.json`);
  if (!(await pathExists(resolved))) {
    throw new Error(`Dataset not found: ${resolved}`);
  }
  return resolved;
}

export async function resolveDatasetAndCorpus(datasetName, datasetPath = "", corpusPath = "") {
  const dataset = await resolveDatasetPath(datasetName, datasetPath);
  if (corpusPath) {
    const corpus = path.resolve(corpusPath);
    if (!(await pathExists(corpus))) throw new Error(`Corpus not found: ${corpus}`);
    return { dataset, corpus };
  }

  const report = await readJson(path.join(DATA_DIR, `${path.basename(dataset, ".json")}.json.report.json`), null);
  if (report && typeof report === "object") {
    for (const key of ["corpus_path", "corpus", "corpus_source"]) {
      const value = String(report[key] ?? "").trim();
      if (value) {
        const candidate = path.resolve(value);
        if (await pathExists(candidate)) return { dataset, corpus: candidate };
      }
    }
  }

  for (const suffix of [".corpus.txt", ".corpus.md"]) {
    const candidate = path.join(DATA_DIR, `${path.basename(dataset, ".json")}${suffix}`);
    if (await pathExists(candidate)) return { dataset, corpus: candidate };
  }
  return { dataset, corpus: null };
}

export function buildFallbackCorpus(rows) {
  const parts = [];
  const seen = new Set();
  for (const row of rows) {
    for (const quote of evidenceQuotesFor(row)) {
      const key = normalizeText(quote);
      if (key && !seen.has(key)) {
        seen.add(key);
        parts.push(quote);
      }
    }
  }
  return parts.join("\n\n");
}

export async function buildSelectedCorpus(rows, datasetName) {
  await ensureDir(TEMP_DOCS_DIR);
  const outPath = path.join(TEMP_DOCS_DIR, `deeplocals_selected_${datasetName}_${Math.floor(Date.now() / 1000)}.md`);
  const parts = [];
  const seenPaths = new Set();

  for (const row of rows) {
    if (!Array.isArray(row.evidence)) continue;
    for (const item of row.evidence) {
      if (!item || typeof item !== "object" || !item.source_file) continue;
      const sourcePath = path.resolve(String(item.source_file));
      if (seenPaths.has(sourcePath) || !(await pathExists(sourcePath))) continue;
      seenPaths.add(sourcePath);
      const text = (await fs.readFile(sourcePath, "utf8").catch(() => "")).trim();
      if (text) parts.push(`# Source: ${path.basename(sourcePath)}\n\n${text}`);
    }
  }

  if (!parts.length) parts.push(buildFallbackCorpus(rows));
  await fs.writeFile(outPath, parts.filter((part) => part.trim()).join("\n\n---\n\n"), "utf8");
  return outPath;
}

export async function buildGeneratedCorpus(rows, datasetName) {
  await ensureDir(TEMP_DOCS_DIR);
  const outPath = path.join(TEMP_DOCS_DIR, `${datasetName}.corpus.generated.md`);
  await fs.writeFile(outPath, buildFallbackCorpus(rows), "utf8");
  return outPath;
}
