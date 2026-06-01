import fs from "node:fs/promises";
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
