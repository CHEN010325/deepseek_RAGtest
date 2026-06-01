#!/usr/bin/env node
import { runPlatformCompare } from "../src/compare.js";
import { runCherryEval } from "../src/evaluation/cherry.js";
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

function usage() {
  console.log(`
RAGEval Forge Node CLI

Usage:
  node bin/rageval.js deeplocals --dataset zh_int_clean [--limit 1]
  node bin/rageval.js cherry --dataset zh_int_clean --knowledge-base-id <id>
  node bin/rageval.js compare --dataset zh_int_clean --cherry-knowledge-base-id <id>

Common options:
  --dataset <name>
  --limit <n>
  --ids <id,id>
  --timeout <seconds>
  --output <path>
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
        deepseekmineApiBase: args.deepseekmineApiBase || args.deeplocalApiBase || "http://127.0.0.1:3335",
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
