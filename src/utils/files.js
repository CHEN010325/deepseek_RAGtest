import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

export async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

export async function readText(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return text.replace(/^\uFEFF/, "");
}

export async function readJson(filePath, fallback = null) {
  try {
    return JSON.parse(await readText(filePath));
  } catch {
    return fallback;
  }
}

export async function writeJson(filePath, payload) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

export function sanitizeFilename(value, fallback = "file") {
  const cleaned = String(value || "")
    .replace(/[\\/]+/gu, "_")
    .replace(/[\x00-\x1f<>:"|?*]+/gu, "_")
    .replace(/\s+/gu, " ")
    .trim();
  return cleaned || fallback;
}

export function sanitizeDatasetName(value, fallback = "custom_rag_eval") {
  const cleaned = String(value || "").trim().replace(/[^A-Za-z0-9_.-]+/gu, "_");
  if (!cleaned || cleaned === "." || cleaned === "..") return fallback;
  return cleaned;
}

export async function uniquePath(filePath) {
  if (!(await pathExists(filePath))) return filePath;
  const parsed = path.parse(filePath);
  for (let index = 1; index < 10000; index += 1) {
    const candidate = path.join(parsed.dir, `${parsed.name}_${index}${parsed.ext}`);
    if (!(await pathExists(candidate))) return candidate;
  }
  throw new Error(`Cannot allocate unique path for ${filePath}`);
}

export async function copyFileUnique(source, targetDir, targetName = "") {
  await ensureDir(targetDir);
  const target = await uniquePath(path.join(targetDir, sanitizeFilename(targetName || path.basename(source))));
  await fs.copyFile(source, target);
  return target;
}

export async function sha256File(filePath) {
  const hash = createHash("sha256");
  const handle = await fs.open(filePath, "r");
  try {
    for await (const chunk of handle.readableWebStream({ type: "bytes" })) {
      hash.update(Buffer.from(chunk));
    }
  } finally {
    await handle.close();
  }
  return hash.digest("hex");
}

export function timestampCompact(date = new Date()) {
  const pad = (value) => String(value).padStart(2, "0");
  return [
    date.getFullYear(),
    pad(date.getMonth() + 1),
    pad(date.getDate()),
    "_",
    pad(date.getHours()),
    pad(date.getMinutes()),
    pad(date.getSeconds())
  ].join("");
}

export function isoTimestamp(date = new Date()) {
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const pad = (value) => String(value).padStart(2, "0");
  const offset = `${sign}${pad(Math.floor(abs / 60))}${pad(abs % 60)}`;
  return `${date.toISOString().replace(/\.\d{3}Z$/, "")}${offset}`;
}
