#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { runPlatformCompare } from "../src/compare.js";
import { runCherryEval } from "../src/evaluation/cherry.js";
import { buildDatasetFromFiles, buildDatasetFromText } from "../src/evaluation/datasetBuilder.js";
import { runDeepLocalsEval } from "../src/evaluation/deeplocals.js";

function parseArgv(argv) {
  const args = { _: [] };
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index];
    if (!item.startsWith("--")) {
      args._.push(item);
      continue;
    }
    const eq = item.indexOf("=");
    const key = item
      .slice(2, eq >= 0 ? eq : undefined)
      .replace(/-([a-z])/gu, (_, char) => char.toUpperCase());
    if (eq >= 0) {
      args[key] = item.slice(eq + 1);
    } else if (argv[index + 1] && !argv[index + 1].startsWith("--")) {
      args[key] = argv[index + 1];
      index += 1;
    } else {
      args[key] = true;
    }
  }
  return args;
}

function numberArg(value, fallback = 0) {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isInteger(parsed) ? parsed : fallback;
}

function floatArg(value, fallback = 0) {
  const parsed = Number.parseFloat(String(value ?? ""));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function boolArg(value, fallback = false) {
  if (value === undefined) return fallback;
  if (typeof value === "boolean") return value;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function listArg(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function usage() {
  console.log(`
RAGEval Forge Node CLI

Usage:
  node bin/rageval.js build --dataset custom_rag_eval --input E:\\docs\\a.pdf,E:\\docs\\b.md
  node bin/rageval.js build --dataset custom_rag_eval --text-file E:\\docs\\source.txt
  node bin/rageval.js deeplocals --dataset zh_int_clean [--limit 1]
  node bin/rageval.js cherry --dataset zh_int_clean --knowledge-base-id <id>
  node bin/rageval.js compare --dataset zh_int_clean --cherry-knowledge-base-id <id>

Common options:
  --dataset <name>
  --limit <n>
  --ids <id,id>
  --timeout <seconds>
  --output <path>
  --target-count <n>
  --questions-per-chunk <n>
`.trim());
}

async function main() {
  const [command, ...rest] = process.argv.slice(2);
  const args = parseArgv(rest);
  if (!command || command === "help" || args.help) {
    usage();
    return;
  }
  const log = (message) => console.log(message);

  if (command === "build") {
    const datasetName = args.dataset || args.datasetName || "custom_rag_eval";
    const targetQuestions = numberArg(args.targetCount, 20);
    const questionsPerChunk = numberArg(args.questionsPerChunk, 3);
    const timeout = numberArg(args.timeout, 300);
    if (args.text || args.textFile) {
      const text = args.text ? String(args.text) : await fs.readFile(path.resolve(String(args.textFile)), "utf8");
      const result = await buildDatasetFromText(
        {
          datasetName,
          sourceName: args.sourceName || (args.textFile ? path.basename(String(args.textFile)) : `${datasetName}.txt`),
          text,
          targetQuestions,
          questionsPerChunk,
          timeout
        },
        (message) => log(`[build] ${message}`)
      );
      log(`[build] 完成 qa_count=${result.qa_count}`);
      log(`[build] dataset=${result.dataset_path}`);
      log(`[build] report=${result.report_path}`);
      return;
    }
    const inputs = listArg(args.input || args.inputs);
    if (!inputs.length) throw new Error("build requires --input <file,...> or --text-file <path>");
    const files = inputs.map((input) => {
      const resolved = path.resolve(input);
      return { path: resolved, originalName: path.basename(resolved) };
    });
    const result = await buildDatasetFromFiles(
      {
        datasetName,
        files,
        useMineru: !boolArg(args.noMineru, false),
        mineruModel: args.mineruModel || "vlm",
        language: args.language || "ch",
        isOcr: !boolArg(args.noOcr, false),
        enableTable: !boolArg(args.noTable, false),
        enableFormula: !boolArg(args.noFormula, false),
        targetQuestions,
        questionsPerChunk,
        timeout
      },
      (message) => log(`[build] ${message}`)
    );
    log(`[build] 完成 qa_count=${result.qa_count}`);
    log(`[build] dataset=${result.dataset_path}`);
    log(`[build] report=${result.report_path}`);
    if (result.assets_manifest) log(`[build] assets=${result.assets_manifest}`);
    return;
  }

  if (command === "deeplocals") {
    const report = await runDeepLocalsEval(
      {
        datasetName: args.dataset,
        datasetPath: args.datasetPath || "",
        corpusPath: args.corpusPath || "",
        apiBase: args.apiBase || "http://127.0.0.1:3335",
        knowledgeLabel: args.knowledgeLabel || "",
        mode: args.mode || "qa",
        limit: numberArg(args.limit, 0),
        ids: args.ids || "",
        output: args.output || "",
        includePrompts: !args.noPrompts,
        useAdaptiveRag: !args.noAdaptiveRag,
        timeout: numberArg(args.timeout, 900)
      },
      (message) => log(`[eval] ${message}`)
    );
    const summary = report.summary;
    log(`[eval] 完成 qa_accuracy=${summary.qa_accuracy.toFixed(4)} (${summary.qa_correct}/${summary.qa_total})`);
    log(`[eval] report=${report.output_path}`);
    return;
  }

  if (command === "cherry") {
    const report = await runCherryEval(
      {
        datasetName: args.dataset,
        datasetPath: args.datasetPath || "",
        apiBase: args.apiBase || "http://127.0.0.1:23333",
        apiKey: args.apiKey || "",
        knowledgeBaseIds: args.knowledgeBaseId || args.knowledgeBaseIds || "",
        model: args.model || "",
        documentCount: numberArg(args.documentCount, 5),
        limit: numberArg(args.limit, 0),
        ids: args.ids || "",
        output: args.output || "",
        timeout: numberArg(args.timeout, 300),
        judgeMode: args.judgeMode || "rule_then_model",
        temperature: floatArg(args.temperature, 0),
        maxTokens: numberArg(args.maxTokens, 1024)
      },
      (message) => log(`[cherry] ${message}`)
    );
    const summary = report.summary;
    log(`[cherry] 完成 qa_accuracy=${summary.qa_accuracy.toFixed(4)} (${summary.qa_correct}/${summary.qa_total})`);
    log(`[cherry] report=${report.output_path}`);
    return;
  }

  if (command === "compare") {
    const summary = await runPlatformCompare(
      {
        dataset: args.dataset || "zh_int_clean",
        limit: numberArg(args.limit, 0),
        ids: args.ids || "",
        timeout: numberArg(args.timeout, 900),
        documentCount: numberArg(args.documentCount, 20),
        deeplocalApiBase: args.deeplocalApiBase || "http://127.0.0.1:3335",
        cherryApiBase: args.cherryApiBase || "http://127.0.0.1:23333",
        cherryApiKey: args.cherryApiKey || "",
        cherryKnowledgeBaseId: args.cherryKnowledgeBaseId || args.knowledgeBaseId || "",
        cherryModel: args.cherryModel || "silicon:deepseek-ai/DeepSeek-V4-Flash",
        deeplocalAnswerProvider: args.deeplocalAnswerProvider || "siliconflow",
        deeplocalAnswerModel: args.deeplocalAnswerModel || "deepseek-ai/DeepSeek-V4-Flash",
        deeplocalAnswerApiUrl: args.deeplocalAnswerApiUrl || "",
        deeplocalAnswerApiKey: args.deeplocalAnswerApiKey || "",
        outputPrefix: args.outputPrefix || "",
        noPrompts: Boolean(args.noPrompts)
      },
      log
    );
    log(`[compare] winner=${summary.winner_by_qa_accuracy}`);
    log(`[compare] summary=${summary.summary_path}`);
    return;
  }

  throw new Error(`Unknown command: ${command}`);
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exitCode = 1;
});
