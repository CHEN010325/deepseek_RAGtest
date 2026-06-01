import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { DATA_DIR, RESULT_ZH_DIR } from "./config.js";
import { runPlatformCompare } from "./compare.js";
import { runCherryEval } from "./evaluation/cherry.js";
import { buildDatasetFromText } from "./evaluation/datasetBuilder.js";
import { runDeepLocalsEval } from "./evaluation/deeplocals.js";
import { ensureDir, pathExists, readJson, readText } from "./utils/files.js";

const jobs = new Map();

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

async function readBodyJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  const text = Buffer.concat(chunks).toString("utf8");
  return text ? JSON.parse(text) : {};
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
    datasets.push({
      name,
      path: filePath,
      rows: await countJsonl(filePath).catch(() => 0),
      has_corpus: (await pathExists(path.join(DATA_DIR, `${name}.corpus.md`))) || (await pathExists(path.join(DATA_DIR, `${name}.corpus.txt`))),
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
    reports.push({
      name: entry.name,
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

function createJob(kind, runner) {
  const id = `${kind}_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;
  const job = {
    id,
    kind,
    status: "running",
    logs: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    result: null,
    error: ""
  };
  jobs.set(id, job);
  const log = (message) => {
    job.logs.push({ at: new Date().toISOString(), message: String(message) });
    if (job.logs.length > 400) job.logs.shift();
    job.updated_at = new Date().toISOString();
  };
  Promise.resolve()
    .then(() => runner(log))
    .then((result) => {
      job.status = "completed";
      job.result = result;
      job.updated_at = new Date().toISOString();
    })
    .catch((error) => {
      job.status = "failed";
      job.error = error.stack || error.message || String(error);
      job.updated_at = new Date().toISOString();
    });
  return job;
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
  if (req.method === "GET" && url.pathname === "/api/datasets") return jsonResponse(res, { datasets: await listDatasets() });
  if (req.method === "GET" && url.pathname === "/api/reports") return jsonResponse(res, { reports: await listReports() });
  if (req.method === "GET" && url.pathname.startsWith("/api/jobs/")) {
    const id = decodeURIComponent(url.pathname.split("/").pop() || "");
    const job = jobs.get(id);
    return jsonResponse(res, job ? { job } : { error: "job not found" }, job ? 200 : 404);
  }
  if (req.method === "POST" && url.pathname === "/api/datasets/build") {
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
    .badge { border-radius:999px; padding:5px 10px; font-size:12px; font-weight:800; background:#e8eef6; color:#334155; }
    .badge.running { background:#dbeafe; color:#1d4ed8; }
    .badge.completed { background:#dcfce7; color:#166534; }
    .badge.failed { background:#fee2e2; color:#991b1b; }
    .muted { color:var(--muted); }
    table { width:100%; border-collapse:collapse; margin-top:10px; }
    th, td { text-align:left; padding:9px 8px; border-bottom:1px solid #edf1f5; vertical-align:top; }
    th { font-size:12px; color:#53657a; }
    code { background:#eef2f6; padding:2px 5px; border-radius:4px; }
    pre { background:#101722; color:#d6e3f3; border-radius:8px; padding:12px; overflow:auto; min-height:260px; max-height:420px; white-space:pre-wrap; }
    .monitor-table { padding:0 22px 22px; }
    .links { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
    .links a { color:#075f58; font-weight:750; text-decoration:none; }
    @media (max-width:1060px) {
      .shell { grid-template-columns:1fr; }
      .app-header { align-items:flex-start; flex-direction:column; }
      .top-pills { justify-content:flex-start; }
    }
    @media (max-width:640px) {
      .shell { padding:14px; }
      .grid, .triple-grid, .workspace-tabs, .metric-strip, .compare-hero, .compare-cards, .secret-grid { grid-template-columns:1fr; }
      .panel-pad, .section-head, .monitor-head, .run-body { padding-left:16px; padding-right:16px; }
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
          <p>粘贴 TXT/Markdown 原文后，Node 版会按证据块生成问题、答案和可核验证据，并输出到 <code>data/</code>。</p>
        </div>
        <div class="panel-pad">
          <div class="metric-strip">
            <div class="metric"><span>运行时</span><strong>Node.js</strong></div>
            <div class="metric"><span>负例</span><strong>不生成</strong></div>
            <div class="metric"><span>输出</span><strong>JSONL</strong></div>
          </div>
        <div class="grid">
          <div><label>数据集名</label><input id="buildName" value="custom_rag_eval" /></div>
          <div><label>来源文件名</label><input id="sourceName" value="manual_source.txt" /></div>
          <div><label>目标题数</label><input id="targetQuestions" type="number" value="20" /></div>
          <div><label>每块题数</label><input id="questionsPerChunk" type="number" value="3" /></div>
        </div>
        <div class="field"><label>原文</label>
        <textarea id="sourceText" placeholder="粘贴 TXT/Markdown/HTML 纯文本内容"></textarea>
        </div>
        <div class="action-row"><button class="primary" onclick="buildDataset()">开始生成数据集</button><span class="muted">使用当前 LLM 配置或环境变量。</span></div>
        </div>
      </div>

      <div class="workflow-panel" data-workspace-panel="eval">
        <div class="section-head">
          <div class="kicker">Real RAG Evaluation</div>
          <h2>选择整个数据集进行真实测评</h2>
          <p>选中数据集后，上传对应 corpus 到新的 DeepLocals 知识库，并用该数据集问答对完成端到端评估。</p>
        </div>
        <div class="panel-pad">
        <div class="grid">
          <div><label>数据集</label><input id="dataset" value="zh_int_clean" /></div>
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
        <span id="badge" class="badge">idle</span>
      </div>
      <div class="run-body">
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
  <script>
    const state = { jobs: new Map(), activeTab: 'build', activeJobId: null };
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
    async function refreshAll() { await Promise.all([refreshDatasets(), refreshReports()]); renderJobs(); }
    async function refreshDatasets() {
      const { datasets } = await api('/api/datasets');
      document.getElementById('datasets').innerHTML = '<tr><th>名称</th><th>题数</th><th>Corpus</th><th>更新时间</th></tr>' + datasets.map(d =>
        '<tr><td><code>' + d.name + '</code></td><td>' + d.rows + '</td><td>' + (d.has_corpus ? '有' : '无') + '</td><td>' + new Date(d.updated_at).toLocaleString() + '</td></tr>'
      ).join('');
    }
    async function refreshReports() {
      const { reports } = await api('/api/reports');
      document.getElementById('reports').innerHTML = '<tr><th>报告</th><th>数据集</th><th>准确率</th><th>正确</th></tr>' + reports.map(r => {
        const acc = r.qa_accuracy == null ? '-' : (r.qa_accuracy * 100).toFixed(2) + '%';
        return '<tr><td><code>' + r.name + '</code></td><td>' + (r.dataset || '-') + '</td><td>' + acc + '</td><td>' + (r.qa_correct ?? '-') + '/' + (r.qa_total ?? '-') + '</td></tr>';
      }).join('');
    }
    async function startJob(path, payload) {
      const { job } = await api(path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
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
        if (job.status !== 'running') { clearInterval(timer); refreshAll(); }
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
      const logs = job ? (job.logs || []).map(x => x.message).join('\n') : '';
      document.getElementById('log').textContent = job?.error || logs || '等待任务...';
      document.getElementById('log').scrollTop = document.getElementById('log').scrollHeight;
      const resultPath = job?.result?.output_path || job?.result?.summary_path || '';
      document.getElementById('links').innerHTML = resultPath ? '<span>输出：</span><code>' + escapeHtml(resultPath) + '</code>' : '';
    }
    function evalPayload() {
      return {
        dataset: val('dataset'), limit: Number(val('limit') || 0), timeout: Number(val('timeout') || 900),
        deeplocalApiBase: val('deepBase'), cherryApiBase: val('cherryBase'), cherryKnowledgeBaseId: val('cherryKb'),
        cherryModel: val('cherryModel'), documentCount: Number(val('documentCount') || 20),
        cherryApiKey: val('cherryKey'), deeplocalAnswerApiKey: val('deepKey')
      };
    }
    function runCompare() { startJob('/api/eval/compare', evalPayload()); }
    function runDeepLocals() { const p = evalPayload(); startJob('/api/eval/deeplocals', { dataset:p.dataset, apiBase:p.deeplocalApiBase, limit:p.limit, timeout:p.timeout }); }
    function runCherry() { const p = evalPayload(); startJob('/api/eval/cherry', { dataset:p.dataset, apiBase:p.cherryApiBase, apiKey:p.cherryApiKey, knowledgeBaseId:p.cherryKnowledgeBaseId, model:p.cherryModel, documentCount:p.documentCount, limit:p.limit, timeout:p.timeout }); }
    function buildDataset() {
      startJob('/api/datasets/build', {
        datasetName: val('buildName'), sourceName: val('sourceName'), targetQuestions: Number(val('targetQuestions') || 20),
        questionsPerChunk: Number(val('questionsPerChunk') || 3), text: val('sourceText')
      });
    }
    function val(id) { return document.getElementById(id).value.trim(); }
    function escapeHtml(s) { return String(s || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    refreshAll();
  </script>
</body>
</html>`;
}
