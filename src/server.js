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
    apiBase: payload.apiBase || payload.deepseekmineApiBase || "http://127.0.0.1:3335",
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
    deepseekmineApiBase: payload.deepseekmineApiBase || payload.deeplocalApiBase || "http://127.0.0.1:3335",
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
  <title>RAGEval Forge Node</title>
  <style>
    :root { color-scheme: light; --bg:#f5f7f9; --panel:#fff; --line:#d9e1ea; --text:#142033; --muted:#607086; --accent:#087a70; --accent2:#1b5fbf; --bad:#b42318; }
    * { box-sizing: border-box; }
    body { margin:0; font:14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--text); background:var(--bg); }
    header { height:64px; display:flex; align-items:center; justify-content:space-between; padding:0 28px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:2; }
    h1 { font-size:20px; margin:0; }
    main { max-width:1180px; margin:0 auto; padding:20px; display:grid; grid-template-columns: 270px 1fr; gap:18px; }
    nav { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; height:max-content; position:sticky; top:84px; }
    nav button { width:100%; text-align:left; border:0; background:transparent; padding:10px 12px; border-radius:6px; font-weight:600; color:var(--muted); cursor:pointer; }
    nav button.active { background:#e7f3f1; color:#055b55; }
    section { display:none; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; }
    section.active { display:block; }
    h2 { margin:0 0 14px; font-size:18px; }
    h3 { margin:20px 0 10px; font-size:14px; text-transform:uppercase; letter-spacing:0; color:#075f58; }
    .grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:12px; }
    label { display:block; font-size:12px; font-weight:700; color:#32445b; margin-bottom:5px; }
    input, textarea, select { width:100%; border:1px solid #bdcad8; border-radius:6px; padding:9px 10px; font:inherit; background:#fff; color:var(--text); }
    textarea { min-height:180px; resize:vertical; }
    button.primary { background:var(--accent); color:#fff; border:0; border-radius:6px; padding:10px 14px; font-weight:800; cursor:pointer; }
    button.secondary { background:#eef4fb; color:#174d91; border:1px solid #bfd2e8; border-radius:6px; padding:9px 12px; font-weight:700; cursor:pointer; }
    .bar { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-top:14px; }
    .muted { color:var(--muted); }
    table { width:100%; border-collapse:collapse; margin-top:10px; }
    th, td { text-align:left; padding:9px 8px; border-bottom:1px solid #edf1f5; vertical-align:top; }
    th { font-size:12px; color:#53657a; }
    code { background:#eef2f6; padding:2px 5px; border-radius:4px; }
    pre { background:#101722; color:#d6e3f3; border-radius:8px; padding:12px; overflow:auto; max-height:360px; }
    .job { border:1px solid var(--line); border-radius:8px; padding:12px; margin-top:12px; background:#fbfcfd; }
    .status-running { color:var(--accent2); font-weight:800; }
    .status-completed { color:var(--accent); font-weight:800; }
    .status-failed { color:var(--bad); font-weight:800; }
  </style>
</head>
<body>
  <header>
    <div><h1>RAGEval Forge</h1><div class="muted">纯 Node.js RAG 测评工作台</div></div>
    <button class="secondary" onclick="refreshAll()">刷新</button>
  </header>
  <main>
    <nav>
      <button class="active" data-tab="overview">概览</button>
      <button data-tab="build">生成数据集</button>
      <button data-tab="eval">平台测评</button>
      <button data-tab="jobs">任务日志</button>
    </nav>
    <div>
      <section id="overview" class="active">
        <h2>数据集</h2>
        <div class="muted">来自 <code>data/*.json</code>，报告来自 <code>result-zh/*.json</code>。</div>
        <table id="datasets"></table>
        <h2 style="margin-top:22px">最近报告</h2>
        <table id="reports"></table>
      </section>
      <section id="build">
        <h2>从文本生成 QA 数据集</h2>
        <div class="grid">
          <div><label>数据集名</label><input id="buildName" value="custom_rag_eval" /></div>
          <div><label>来源文件名</label><input id="sourceName" value="manual_source.txt" /></div>
          <div><label>目标题数</label><input id="targetQuestions" type="number" value="20" /></div>
          <div><label>每块题数</label><input id="questionsPerChunk" type="number" value="3" /></div>
        </div>
        <h3>原文</h3>
        <textarea id="sourceText" placeholder="粘贴 TXT/Markdown/HTML 纯文本内容"></textarea>
        <div class="bar"><button class="primary" onclick="buildDataset()">开始生成</button><span class="muted">使用当前 LLM 配置或环境变量。</span></div>
      </section>
      <section id="eval">
        <h2>DeepLocals vs Cherry Studio</h2>
        <div class="grid">
          <div><label>数据集</label><input id="dataset" value="zh_int_clean" /></div>
          <div><label>题目上限，0 为全量</label><input id="limit" type="number" value="0" /></div>
          <div><label>DeepLocals 地址</label><input id="deepBase" value="http://127.0.0.1:3335" /></div>
          <div><label>Cherry API 地址</label><input id="cherryBase" value="http://127.0.0.1:23333" /></div>
          <div><label>Cherry Knowledge Base ID</label><input id="cherryKb" /></div>
          <div><label>Cherry 模型</label><input id="cherryModel" value="silicon:deepseek-ai/DeepSeek-V4-Flash" /></div>
          <div><label>Cherry 检索块数</label><input id="documentCount" type="number" value="20" /></div>
          <div><label>超时秒数</label><input id="timeout" type="number" value="900" /></div>
        </div>
        <h3>密钥</h3>
        <div class="grid">
          <div><label>Cherry API Key，可留空使用环境变量</label><input id="cherryKey" type="password" /></div>
          <div><label>DeepLocals 回答模型 Key，可留空使用环境变量</label><input id="deepKey" type="password" /></div>
        </div>
        <div class="bar">
          <button class="primary" onclick="runCompare()">并发横评</button>
          <button class="secondary" onclick="runDeepLocals()">只测 DeepLocals</button>
          <button class="secondary" onclick="runCherry()">只测 Cherry</button>
        </div>
      </section>
      <section id="jobs">
        <h2>任务日志</h2>
        <div id="jobList"></div>
      </section>
    </div>
  </main>
  <script>
    const state = { jobs: new Map(), activeTab: 'overview' };
    document.querySelectorAll('nav button').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
    function switchTab(tab) {
      state.activeTab = tab;
      document.querySelectorAll('nav button').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tab));
      document.querySelectorAll('section').forEach(sec => sec.classList.toggle('active', sec.id === tab));
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
      switchTab('jobs');
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
      document.getElementById('jobList').innerHTML = jobs.map(job => {
        const logs = (job.logs || []).map(x => x.message).join('\n');
        const resultPath = job.result?.output_path || job.result?.summary_path || '';
        return '<div class="job"><div><b>' + job.kind + '</b> <span class="status-' + job.status + '">' + job.status + '</span> <code>' + job.id + '</code></div>' +
          (resultPath ? '<div>输出：<code>' + resultPath + '</code></div>' : '') +
          (job.error ? '<pre>' + escapeHtml(job.error) + '</pre>' : '<pre>' + escapeHtml(logs || '等待日志...') + '</pre>') +
          '</div>';
      }).join('') || '<div class="muted">暂无任务</div>';
    }
    function evalPayload() {
      return {
        dataset: val('dataset'), limit: Number(val('limit') || 0), timeout: Number(val('timeout') || 900),
        deepseekmineApiBase: val('deepBase'), cherryApiBase: val('cherryBase'), cherryKnowledgeBaseId: val('cherryKb'),
        cherryModel: val('cherryModel'), documentCount: Number(val('documentCount') || 20),
        cherryApiKey: val('cherryKey'), deeplocalAnswerApiKey: val('deepKey')
      };
    }
    function runCompare() { startJob('/api/eval/compare', evalPayload()); }
    function runDeepLocals() { const p = evalPayload(); startJob('/api/eval/deeplocals', { dataset:p.dataset, apiBase:p.deepseekmineApiBase, limit:p.limit, timeout:p.timeout }); }
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
