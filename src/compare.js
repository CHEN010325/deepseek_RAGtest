import fs from "node:fs/promises";
import path from "node:path";
import { RESULT_ZH_DIR } from "./config.js";
import { defaultProviderConfig } from "./llm.js";
import { ensureDir, isoTimestamp, timestampCompact, writeJson } from "./utils/files.js";
import { runCherryEval } from "./evaluation/cherry.js";
import { runDeepLocalsEval } from "./evaluation/deeplocals.js";

function pct(value) {
  const number = Number(value ?? 0);
  return Number.isFinite(number) ? Math.round(number * 10000) / 100 : 0;
}

function rowFor(platform, report) {
  const summary = report.summary ?? {};
  return {
    platform,
    dataset: report.dataset_name ?? report.dataset ?? "",
    qa_accuracy_percent: pct(summary.qa_accuracy),
    qa_correct: summary.qa_correct ?? 0,
    qa_total: summary.qa_total ?? summary.total ?? 0,
    evidence_hit_rate_percent: pct(summary.evidence_hit_rate),
    evidence_recall_percent: pct(summary.evidence_recall),
    mrr: Math.round(Number(summary.mrr ?? 0) * 10000) / 10000,
    avg_retrieved_count: Math.round(Number(summary.avg_retrieved_count ?? 0) * 100) / 100,
    answer_model: report.answer_model ?? report.model ?? "",
    report_path: report.output_path ?? "",
    completed: Boolean(report.completed),
    stopped_reason: report.stopped_reason ?? ""
  };
}

async function writeCsv(filePath, rows) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const escapeCell = (value) => {
    const text = String(value ?? "");
    return /[",\n]/u.test(text) ? `"${text.replace(/"/gu, "\"\"")}"` : text;
  };
  const content = [headers.join(","), ...rows.map((row) => headers.map((header) => escapeCell(row[header])).join(","))].join("\n");
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, `\uFEFF${content}\n`, "utf8");
}

export async function runPlatformCompare(options, progress = () => {}) {
  const stamp = timestampCompact();
  const prefix = options.outputPrefix || `platform_compare_${options.dataset}_${stamp}`;
  const deepReportPath = path.join(RESULT_ZH_DIR, `${prefix}_deeplocal.json`);
  const cherryReportPath = path.join(RESULT_ZH_DIR, `${prefix}_cherry.json`);
  await ensureDir(RESULT_ZH_DIR);

  let answerConfig = null;
  if (options.deeplocalAnswerProvider) {
    answerConfig = await defaultProviderConfig(options.deeplocalAnswerProvider);
    if (options.deeplocalAnswerApiUrl) answerConfig.api_url = options.deeplocalAnswerApiUrl;
    if (options.deeplocalAnswerModel) {
      answerConfig.model = options.deeplocalAnswerModel;
      if (!answerConfig.models.includes(answerConfig.model)) answerConfig.models.unshift(answerConfig.model);
    }
    if (options.deeplocalAnswerApiKey) answerConfig.api_key = options.deeplocalAnswerApiKey;
    else if (answerConfig.provider_id === "siliconflow" && process.env.SILICONFLOW_API_KEY) {
      answerConfig.api_key = process.env.SILICONFLOW_API_KEY.trim();
    }
    if (answerConfig.provider_id === "ollama") answerConfig.api_key = "";
    if (answerConfig.provider_id !== "ollama" && !answerConfig.api_key) {
      throw new Error(`${answerConfig.provider_name} API key is missing for DeepLocals answer model.`);
    }
  }

  progress(`[compare] dataset=${options.dataset} limit=${options.limit || "all"} ids=${options.ids || "-"}`);
  progress("[compare] running DeepLocals and Cherry Studio in parallel...");
  const [deepReport, cherryReport] = await Promise.all([
    runDeepLocalsEval(
      {
        datasetName: options.dataset,
        apiBase: options.deepseekmineApiBase,
        mode: "qa",
        limit: options.limit,
        ids: options.ids,
        output: deepReportPath,
        includePrompts: !options.noPrompts,
        timeout: options.timeout,
        answerConfig
      },
      (message) => progress(`[deeplocal] ${message}`)
    ),
    runCherryEval(
      {
        datasetName: options.dataset,
        apiBase: options.cherryApiBase,
        apiKey: options.cherryApiKey || process.env.CHERRY_API_KEY || "",
        knowledgeBaseIds: options.cherryKnowledgeBaseId,
        model: options.cherryModel,
        documentCount: options.documentCount,
        limit: options.limit,
        ids: options.ids,
        output: cherryReportPath,
        timeout: options.timeout,
        judgeMode: "rule_then_model"
      },
      (message) => progress(`[cherry] ${message}`)
    )
  ]);

  const rows = [rowFor("DeepLocals", deepReport), rowFor("Cherry Studio", cherryReport)];
  const winner = rows.reduce((best, row) => (row.qa_accuracy_percent > best.qa_accuracy_percent ? row : best), rows[0]);
  const summaryPath = path.join(RESULT_ZH_DIR, `${prefix}_summary.json`);
  const csvPath = path.join(RESULT_ZH_DIR, `${prefix}_summary.csv`);
  const payload = {
    created_at: isoTimestamp(),
    dataset: options.dataset,
    limit: options.limit,
    ids: options.ids,
    winner_by_qa_accuracy: winner.platform,
    rows
  };
  await writeJson(summaryPath, payload);
  await writeCsv(csvPath, rows);
  return { ...payload, summary_path: summaryPath, csv_path: csvPath };
}
