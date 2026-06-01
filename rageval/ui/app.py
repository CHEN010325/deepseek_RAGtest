"""
Local web UI for generating RAG evaluation datasets.

Run:
    python E:\\RAG\\rag_dataset_ui.py --host 127.0.0.1 --port 7861
"""

from __future__ import annotations

import argparse
import cgi
import json
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote, urlparse

from rageval.builder.dataset_builder import build_dataset, write_jsonl
from rageval.config import (
    APP_ROOT,
    ASSETS_DIR,
    DATA_DIR,
    MINERU_CACHE_DIR,
    MINERU_KEY_FILE,
    RESULT_ZH_DIR,
    RUNS_DIR,
)
from rageval.deepseekmine.compat_import import write_chunk_sidecar
from rageval.evaluation import deepseekmine_rag_eval as deep_eval
from rageval.evaluation.cherry_rag_eval import CherryEvalOptions, run_eval as run_cherry_eval
from rageval.evaluation.deepseekmine_rag_eval import EvalOptions, run_eval
from rageval.llm import (
    config_from_form,
    default_provider_config,
    list_provider_models,
    load_ollama_context_config,
    load_llm_config,
    masked_config,
    save_ollama_context_config,
    provider_presets_payload,
    save_llm_config,
)
from rageval.mineru.ocr import MineruOptions, mineru_supported_paths, process_with_mineru


RESULT_DIR = RESULT_ZH_DIR
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


class JobCancelled(RuntimeError):
    pass


def preserve_saved_api_key(config):
    if config.provider_id == "ollama":
        config.api_key = ""
        return config
    if config.api_key:
        return config
    saved = load_llm_config()
    if saved.provider_id == config.provider_id and saved.api_key:
        config.api_key = saved.api_key
    return config


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG 数据集生成器</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #182033;
      --muted: #647086;
      --line: #d9deea;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --ok: #087443;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 0 28px;
    }
    header h1 {
      font-size: 20px;
      margin: 0;
      font-weight: 650;
    }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 520px) minmax(420px, 1fr);
      gap: 20px;
      padding: 20px;
      max-width: 1440px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    h2 {
      margin: 0 0 16px;
      font-size: 16px;
      font-weight: 650;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    input, select {
      width: 100%;
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    input[type="file"] {
      height: auto;
      padding: 10px;
    }
    input[type="checkbox"] {
      width: 16px;
      height: 16px;
      vertical-align: middle;
      margin: 0 8px 0 0;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .field { margin-bottom: 12px; }
    .checkrow {
      display: flex;
      align-items: center;
      min-height: 32px;
      color: var(--text);
      font-size: 14px;
    }
    button {
      height: 40px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 650;
      padding: 0 16px;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .muted { color: var(--muted); font-size: 13px; }
    .status {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 14px;
    }
    .badge {
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      background: #eef2f7;
      color: var(--muted);
    }
    .badge.done { background: #e8f6ef; color: var(--ok); }
    .badge.error { background: #fdebea; color: var(--danger); }
    progress {
      width: 100%;
      height: 12px;
      accent-color: var(--accent);
    }
    pre {
      min-height: 360px;
      max-height: 620px;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
    }
    .links a {
      display: inline-flex;
      align-items: center;
      height: 34px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--accent-strong);
      text-decoration: none;
      margin: 0 8px 8px 0;
      background: #fff;
      font-size: 13px;
    }
    .divider {
      height: 1px;
      background: var(--line);
      margin: 22px 0;
    }
    .secondary {
      background: #eef2f7;
      color: var(--text);
    }
    .secondary:hover { background: #e1e7f0; }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header><h1>RAG 数据集生成器</h1></header>
  <main>
    <section>
      <h2>输入与解析</h2>
      <form id="job-form">
        <div class="field">
          <label>文档文件</label>
          <input name="files" type="file" multiple required />
        </div>
        <div class="field">
          <label>输出数据集名</label>
          <input id="build-dataset-name" name="dataset_name" value="custom_rag_eval" pattern="[A-Za-z0-9_.-]+" />
        </div>
        <div class="field">
          <label>解析服务</label>
          <input value="MinerU / 本地固定 Token" disabled />
        </div>
        <div class="grid">
          <div class="field">
            <label>MinerU 模型</label>
            <select name="mineru_model">
              <option value="vlm">vlm</option>
              <option value="pipeline">pipeline</option>
            </select>
          </div>
          <div class="field">
            <label>语言</label>
            <input name="language" value="ch" />
          </div>
        </div>
        <div class="grid">
          <label class="checkrow"><input name="use_mineru" type="checkbox" checked />使用 MinerU OCR/解析</label>
          <label class="checkrow"><input name="is_ocr" type="checkbox" checked />启用 OCR</label>
          <label class="checkrow"><input name="enable_table" type="checkbox" checked />表格识别</label>
          <label class="checkrow"><input name="enable_formula" type="checkbox" checked />公式识别</label>
        </div>

        <h2 style="margin-top:22px">问答生成</h2>
        <div class="field">
          <label>LLM 后端</label>
          <input value="MiMo / MiMo-V2.5-Pro" disabled />
        </div>
        <div class="grid">
          <div class="field">
            <label>目标问题数</label>
            <input name="target_count" type="number" min="1" max="5000" value="100" />
          </div>
          <div class="field">
            <label>每文件上限</label>
            <input name="max_per_source" type="number" min="1" max="5000" value="100" />
          </div>
          <div class="field">
            <label>每块生成数</label>
            <input value="1" disabled />
          </div>
        </div>
        <button id="submit-btn" type="submit">开始生成</button>
        <p class="muted">OCR/解析固定使用本地 MinerU Token；问答生成固定使用 MiMo-V2.5-Pro；输出仍会做答案和证据的硬校验。</p>
      </form>

      <div class="divider"></div>
      <h2>真实 RAG 测评</h2>
      <form id="eval-form">
        <div class="field">
          <label>选择测评数据集</label>
          <select id="eval-dataset-select" name="dataset_name"></select>
        </div>
        <div class="grid">
          <div class="field">
            <label>DeepLocals 地址</label>
            <input name="api_base" value="http://127.0.0.1:3335" />
          </div>
          <div class="field">
            <label>知识库策略</label>
            <input value="每次自动新建隔离知识库" disabled />
          </div>
          <div class="field">
            <label>测评模式</label>
            <select name="mode">
              <option value="qa">检索 + 问答</option>
              <option value="retrieval">只测检索</option>
            </select>
          </div>
        </div>
        <button id="eval-btn" type="submit">运行真实测评</button>
        <p class="muted">会把整个数据集对应 corpus 上传到一个全新的 DeepLocals 知识库，逐条调用 /api/search 获取完整 result_prompt，再生成最终答案并和标准答案、证据对齐评分。</p>
      </form>
      </div>
    </section>
    <section>
      <div class="status">
        <div>
          <h2 style="margin-bottom:4px">任务状态</h2>
          <div id="job-id" class="muted">尚未开始</div>
        </div>
        <div class="monitor-actions">
          <button id="cancel-btn" class="secondary danger-button" type="button" disabled>取消任务</button>
          <span id="badge" class="badge">idle</span>
        </div>
      </div>
      <progress id="progress" value="0" max="100"></progress>
      <div class="links" id="links" style="margin-top:14px"></div>
      <pre id="log"></pre>
    </section>
  </main>
  <script>
    const form = document.getElementById('job-form');
    const evalForm = document.getElementById('eval-form');
    const compareForm = document.getElementById('compare-form');
    const workspaceTabs = Array.from(document.querySelectorAll('[data-workspace-tab]'));
    const workflowPanels = Array.from(document.querySelectorAll('[data-workspace-panel]'));
    const llmConfigForm = document.getElementById('llm-config-form');
    const apiSettingsButton = document.getElementById('api-settings-button');
    const apiSettingsModal = document.getElementById('api-settings-modal');
    const apiSettingsClose = document.getElementById('api-settings-close');
    const llmProvider = document.getElementById('llm-provider');
    const providerList = document.getElementById('provider-list');
    const providerSummaryIcon = document.getElementById('provider-summary-icon');
    const providerSummaryTitle = document.getElementById('provider-summary-title');
    const providerSummarySubtitle = document.getElementById('provider-summary-subtitle');
    const customProviderButton = document.getElementById('custom-provider-button');
    const llmEnabled = document.getElementById('llm-enabled');
    const apiKeyField = document.getElementById('api-key-field');
    const llmModel = document.getElementById('llm-model');
    const modelChoiceList = document.getElementById('model-choice-list');
    const modelChoiceHint = document.getElementById('model-choice-hint');
    const llmConnectionStatus = document.getElementById('llm-connection-status');
    const llmConnectionMark = document.getElementById('llm-connection-mark');
    const llmConnectionTitle = document.getElementById('llm-connection-title');
    const llmConnectionDetail = document.getElementById('llm-connection-detail');
    const ollamaContextPanel = document.getElementById('ollama-context-panel');
    const ollamaContextSlider = document.getElementById('ollama-context-slider');
    const ollamaContextValue = document.getElementById('ollama-context-value');
    const llmApiUrl = document.getElementById('llm-api-url');
    const llmApiKey = document.getElementById('llm-api-key');
    const llmAuthType = document.getElementById('llm-auth-type');
    const llmMaxTokens = document.getElementById('llm-max-tokens');
    const llmTemperature = document.getElementById('llm-temperature');
    const llmTopP = document.getElementById('llm-top-p');
    const llmEnableThinking = document.getElementById('llm-enable-thinking');
    const llmConfigStatus = document.getElementById('llm-config-status');
    const llmTestButton = document.getElementById('llm-test-btn');
    const providerMetric = document.getElementById('provider-metric');
    const modelMetric = document.getElementById('model-metric');
    const buildDatasetName = document.getElementById('build-dataset-name');
    const evalDatasetSelect = document.getElementById('eval-dataset-select');
    const button = document.getElementById('submit-btn');
    const evalButton = document.getElementById('eval-btn');
    const badge = document.getElementById('badge');
    const log = document.getElementById('log');
    const progress = document.getElementById('progress');
    const links = document.getElementById('links');
    const jobIdEl = document.getElementById('job-id');
    let pollTimer = null;

    async function refreshDatasets(preferredName) {
      const res = await fetch('/api/datasets');
      const payload = await res.json();
      const datasets = payload.datasets || [];
      evalDatasetSelect.innerHTML = '';
      if (!datasets.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '暂无可测数据集';
        evalDatasetSelect.appendChild(option);
        return;
      }
      for (const item of datasets) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = item.has_corpus ? `${item.name} (${item.rows}题)` : `${item.name} (${item.rows}题，缺 corpus)`;
        evalDatasetSelect.appendChild(option);
      }
      const wanted = preferredName || buildDatasetName.value.trim();
      if (wanted && datasets.some((item) => item.name === wanted)) {
        evalDatasetSelect.value = wanted;
      }
    }

    function setBadge(status) {
      badge.textContent = status || 'idle';
      badge.className = 'badge' + (status === 'done' ? ' done' : status === 'error' ? ' error' : '');
    }

    async function poll(jobId) {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      setBadge(job.status);
      progress.value = job.progress || 0;
      log.textContent = (job.logs || []).join('\n');
      log.scrollTop = log.scrollHeight;
      links.innerHTML = '';
      if (job.output_url) links.innerHTML += `<a href="${job.output_url}">下载 JSONL</a>`;
      if (job.corpus_url) links.innerHTML += `<a href="${job.corpus_url}">下载 Corpus</a>`;
      if (job.report_url) links.innerHTML += `<a href="${job.report_url}">下载报告</a>`;
      if (job.assets_manifest) links.innerHTML += `<a href="/download?path=${encodeURIComponent(job.assets_manifest)}">资产 Manifest</a>`;
      if (job.status === 'done' || job.status === 'error') {
        button.disabled = false;
        evalButton.disabled = false;
        clearInterval(pollTimer);
        if (job.status === 'done' && job.output_path) {
          refreshDatasets(buildDatasetName.value.trim());
        }
      }
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      log.textContent = '';
      links.innerHTML = '';
      progress.value = 0;
      setBadge('queued');
      const data = new FormData(form);
      const res = await fetch('/api/jobs', { method: 'POST', body: data });
      const payload = await res.json();
      if (!res.ok) {
        setBadge('error');
        button.disabled = false;
        log.textContent = payload.error || '提交失败';
        return;
      }
      jobIdEl.textContent = payload.job_id;
      pollTimer = setInterval(() => poll(payload.job_id), 1200);
      poll(payload.job_id);
    });

    buildDatasetName.addEventListener('input', () => {
      const wanted = buildDatasetName.value.trim();
      if ([...evalDatasetSelect.options].some((option) => option.value === wanted)) {
        evalDatasetSelect.value = wanted;
      }
    });

    evalForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      button.disabled = true;
      evalButton.disabled = true;
      log.textContent = '';
      links.innerHTML = '';
      progress.value = 0;
      setBadge('queued');
      const data = new FormData(evalForm);
      const res = await fetch('/api/eval-jobs', { method: 'POST', body: data });
      const payload = await res.json();
      if (!res.ok) {
        setBadge('error');
        button.disabled = false;
        evalButton.disabled = false;
        log.textContent = payload.error || '提交失败';
        return;
      }
      jobIdEl.textContent = payload.job_id;
      pollTimer = setInterval(() => poll(payload.job_id), 1200);
      poll(payload.job_id);
    });

    refreshDatasets('custom_rag_eval');
  </script>
</body>
</html>
"""


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAGEval Forge</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --surface: #fff;
      --surface-2: #f8fafc;
      --text: #111827;
      --muted: #64748b;
      --line: #d9e0e8;
      --line-strong: #c7d0dc;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --blue: #1d4ed8;
      --amber: #b45309;
      --danger: #b42318;
      --ok: #087443;
      --shadow: 0 18px 48px rgba(15, 23, 42, .08);
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Inter", "Segoe UI", "Microsoft YaHei", system-ui, sans-serif;
      background: linear-gradient(180deg, #edf2f7 0, #f4f6f8 340px), var(--bg);
      color: var(--text);
    }
    button, input, select { font: inherit; letter-spacing: 0; }
    .app-header {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      padding: 22px 32px;
      border-bottom: 1px solid rgba(199, 208, 220, .85);
      background: rgba(255, 255, 255, .88);
      backdrop-filter: blur(14px);
    }
    .brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #102a43;
      color: #fff;
      font-weight: 760;
      box-shadow: inset 0 -1px 0 rgba(255,255,255,.16);
    }
    .brand h1 { margin: 0; font-size: 19px; line-height: 1.2; font-weight: 720; }
    .brand p { margin: 3px 0 0; color: var(--muted); font-size: 13px; line-height: 1.35; }
    .top-pills { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .settings-button {
      height: 34px;
      padding: 0 12px;
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 13px;
      font-weight: 760;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      height: 30px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: #334155;
      font-size: 12px;
      white-space: nowrap;
    }
    .dot { width: 7px; height: 7px; border-radius: 999px; background: var(--accent); }
    .shell {
      display: grid;
      grid-template-columns: minmax(420px, 560px) minmax(460px, 1fr);
      gap: 22px;
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }
    section {
      background: rgba(255, 255, 255, .96);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .workspace, .monitor { padding: 0; overflow: hidden; }
    .panel-pad { padding: 22px; }
    .section-head {
      padding: 22px 22px 18px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fff, #fbfcfe);
    }
    .kicker {
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 740;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    h2 { margin: 0; font-size: 17px; line-height: 1.3; font-weight: 720; }
    .section-head p, .helper, .muted { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .section-head p { margin: 8px 0 0; }
    label {
      display: block;
      font-size: 12px;
      font-weight: 650;
      color: #475569;
      margin: 0 0 7px;
    }
    input, select {
      width: 100%;
      height: 40px;
      border: 1px solid var(--line-strong);
      border-radius: 6px;
      padding: 0 11px;
      background: #fff;
      color: var(--text);
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease, background .16s ease;
    }
    input:focus, select:focus, .picker-button:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .13);
    }
    input:disabled { color: #64748b; background: #f1f5f9; }
    .field { margin-bottom: 14px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .triple-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .workspace-tabs {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #f8fafc;
    }
    .workspace-tab {
      height: 40px;
      border-color: var(--line);
      background: #fff;
      color: #334155;
      font-size: 13px;
    }
    .workspace-tab.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    .workflow-panel { display: none; }
    .workflow-panel.active { display: block; }
    .file-drop {
      display: block;
      border: 1px dashed #aab6c5;
      border-radius: var(--radius);
      background: #f8fafc;
      padding: 18px;
      cursor: pointer;
      transition: border-color .16s ease, background .16s ease;
    }
    .file-drop:hover { border-color: var(--accent); background: #f5fbfa; }
    .file-drop input { width: 100%; height: auto; border: 0; padding: 0; margin-top: 12px; background: transparent; }
    .file-title { display: block; font-size: 14px; font-weight: 720; color: #1f2937; }
    .file-subtitle { display: block; margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.5; }
    .toggle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 4px 0 18px; }
    .checkrow {
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 38px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdff;
      color: #334155;
      font-size: 13px;
      margin: 0;
    }
    .checkrow input { width: 16px; height: 16px; margin: 0; accent-color: var(--accent); }
    .block-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding-top: 20px;
      margin-top: 18px;
      border-top: 1px solid var(--line);
    }
    .metric-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0 16px; }
    .metric { border: 1px solid var(--line); border-radius: 6px; padding: 10px; background: #fbfdff; }
    .metric span { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
    .metric strong { display: block; font-size: 16px; line-height: 1.2; }
    .compare-console { display: grid; gap: 14px; }
    .compare-hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fbfdff;
    }
    .compare-hero h2 { margin-top: 4px; }
    .compare-hero p { margin: 6px 0 0; color: var(--muted); font-size: 13px; line-height: 1.5; }
    .compare-run-button { min-width: 144px; }
    .compare-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .compare-card {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      background: #fff;
    }
    .compare-card span { display: block; font-size: 11px; font-weight: 800; color: var(--accent); text-transform: uppercase; margin-bottom: 6px; }
    .compare-card strong { display: block; font-size: 14px; line-height: 1.35; color: var(--text); }
    .compare-card small { display: block; margin-top: 6px; color: var(--muted); line-height: 1.35; }
    .compare-controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .secret-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .advanced-box {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }
    .advanced-box summary {
      cursor: pointer;
      padding: 11px 12px;
      color: #334155;
      font-size: 13px;
      font-weight: 720;
      list-style: none;
    }
    .advanced-box summary::-webkit-details-marker { display: none; }
    .advanced-body { padding: 0 12px 12px; }
    #compare-form > .grid .field:nth-child(-n+7) { display: none; }
    #compare-form > .grid { grid-template-columns: 1fr 1fr; }
    #compare-form > .grid .field { margin-bottom: 0; }
    #compare-form > .action-row { display: none; }
    #compare-form > .field { margin-bottom: 0; }
    button {
      height: 42px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 720;
      padding: 0 16px;
      cursor: pointer;
      transition: transform .12s ease, background .16s ease, border-color .16s ease, opacity .16s ease;
    }
    button:hover { background: var(--accent-strong); }
    button:active { transform: translateY(1px); }
    button:disabled { opacity: .48; cursor: not-allowed; transform: none; }
    .secondary { background: #fff; color: #1f2937; border-color: var(--line-strong); }
    .secondary:hover { background: #f8fafc; }
    .action-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    .dataset-picker { position: relative; margin-bottom: 14px; }
    .picker-button {
      width: 100%;
      min-height: 66px;
      height: auto;
      padding: 12px 14px;
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      text-align: left;
    }
    .picker-title {
      display: block;
      font-size: 15px;
      font-weight: 720;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .picker-meta { display: block; margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.4; }
    .chevron { color: var(--muted); font-size: 18px; padding-left: 16px; }
    .picker-menu {
      position: absolute;
      left: 0;
      right: 0;
      top: calc(100% + 8px);
      z-index: 30;
      border: 1px solid var(--line-strong);
      border-radius: var(--radius);
      background: #fff;
      box-shadow: 0 22px 48px rgba(15, 23, 42, .16);
      padding: 10px;
      display: none;
    }
    .picker-menu.open { display: block; }
    .picker-search { height: 36px; margin-bottom: 9px; }
    .dataset-list { max-height: 278px; overflow: auto; display: grid; gap: 7px; }
    .dataset-option {
      width: 100%;
      min-height: 58px;
      height: auto;
      padding: 10px 11px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      text-align: left;
      align-items: center;
    }
    .dataset-option:hover { background: #f8fafc; border-color: #9fb0c3; }
    .dataset-option.selected { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(15, 118, 110, .12); }
    .dataset-option.unavailable { opacity: .62; cursor: not-allowed; background: #f8fafc; }
    .dataset-main {
      min-width: 0;
      border: 0;
      background: transparent;
      padding: 0;
      color: inherit;
      text-align: left;
      cursor: pointer;
    }
    .dataset-option.unavailable .dataset-main { cursor: not-allowed; }
    .dataset-name { display: block; font-size: 13px; font-weight: 720; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dataset-detail { display: block; margin-top: 4px; font-size: 12px; color: var(--muted); line-height: 1.35; }
    .dataset-actions { display: flex; align-items: center; gap: 8px; }
    .icon-danger {
      width: 30px;
      height: 30px;
      border-radius: 6px;
      border: 1px solid #fecaca;
      background: #fff7f7;
      color: #b42318;
      padding: 0;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
    }
    .icon-danger:hover { background: #fee2e2; border-color: #fca5a5; }
    .tag {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
      background: #e8f6ef;
      color: var(--ok);
    }
    .tag.warn { background: #fff7ed; color: var(--amber); }
    .empty-state {
      padding: 16px;
      border: 1px dashed var(--line-strong);
      border-radius: 6px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      background: #f8fafc;
    }
    .strategy {
      height: 40px;
      display: flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 11px;
      background: #f8fafc;
      color: #475569;
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .monitor-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      padding: 22px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    .job-label { margin-top: 6px; color: var(--muted); font-size: 13px; }
    .badge {
      display: inline-flex;
      align-items: center;
      height: 30px;
      border-radius: 6px;
      padding: 0 10px;
      font-size: 12px;
      font-weight: 760;
      background: #eef2f7;
      color: #475569;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .badge.done { background: #e8f6ef; color: var(--ok); }
    .badge.error { background: #fdebea; color: var(--danger); }
    .monitor-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .danger-button { color: var(--danger); border-color: #f1b8b2; }
    .danger-button:hover { background: #fff5f4; }
    .run-body { padding: 18px 22px 22px; }
    progress { width: 100%; height: 10px; border: 0; border-radius: 999px; overflow: hidden; accent-color: var(--accent); }
    progress::-webkit-progress-bar { background: #e7edf4; border-radius: 999px; }
    progress::-webkit-progress-value { background: var(--accent); border-radius: 999px; }
    .progress-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }
    .links { min-height: 36px; margin-top: 14px; }
    .links a {
      display: inline-flex;
      align-items: center;
      height: 34px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--accent-strong);
      text-decoration: none;
      margin: 0 8px 8px 0;
      background: #fff;
      font-size: 13px;
      font-weight: 650;
    }
    pre {
      min-height: 480px;
      max-height: 680px;
      overflow: auto;
      margin: 8px 0 0;
      background: #10151f;
      color: #e5edf5;
      border: 1px solid #1f2937;
      border-radius: var(--radius);
      padding: 15px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.55;
    }
    .modal-shell {
      position: fixed;
      inset: 0;
      z-index: 100;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 22px;
      background: rgba(15, 23, 42, .42);
    }
    .modal-shell.open { display: flex; }
    .modal-panel {
      width: min(1024px, calc(100vw - 44px));
      max-height: min(860px, calc(100vh - 44px));
      overflow: hidden;
      background: #fff;
      border: 1px solid var(--line-strong);
      border-radius: var(--radius);
      box-shadow: 0 28px 70px rgba(15, 23, 42, .28);
    }
    .modal-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      padding: 22px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }
    .modal-head h2 { margin: 4px 0 6px; font-size: 20px; }
    .modal-head p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .modal-close {
      width: 36px;
      height: 36px;
      padding: 0;
      border-radius: 6px;
      flex: 0 0 auto;
    }
    .api-config-layout {
      display: grid;
      grid-template-columns: 286px minmax(0, 1fr);
      min-height: 500px;
      max-height: calc(100vh - 130px);
    }
    .api-provider-pane {
      border-right: 1px solid var(--line);
      background: #fbfdff;
      padding: 20px 16px;
      overflow: auto;
    }
    .api-pane-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }
    .api-pane-head strong { font-size: 14px; }
    .tiny-button {
      height: 28px;
      padding: 0 10px;
      font-size: 12px;
      font-weight: 650;
      border-color: #bfdbfe;
      color: #1d4ed8;
      background: #fff;
    }
    .provider-list { display: grid; gap: 10px; }
    .provider-card {
      width: 100%;
      min-height: 72px;
      height: auto;
      padding: 12px;
      display: grid;
      grid-template-columns: 38px 1fr auto;
      align-items: center;
      gap: 12px;
      text-align: left;
      color: var(--text);
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .provider-card:hover { background: #f8fbff; border-color: #b7cffd; }
    .provider-card.active {
      background: #eff6ff;
      border-color: #93c5fd;
      box-shadow: 0 0 0 2px rgba(59, 130, 246, .12);
    }
    .provider-icon {
      width: 38px;
      height: 38px;
      border-radius: 9px;
      display: grid;
      place-items: center;
      background: #eef2ff;
      color: #4f46e5;
      font-weight: 820;
      font-size: 16px;
      box-shadow: 0 8px 18px rgba(37, 99, 235, .12);
    }
    .provider-icon.siliconflow { background: #f1eaff; color: #7c3aed; }
    .provider-icon.deepseek { background: #edf2ff; color: #2563eb; }
    .provider-icon.qwen { background: #e8fbff; color: #0891b2; }
    .provider-icon.mimo { background: #fff2e8; color: #ea580c; }
    .provider-icon.ollama { background: #ecfdf5; color: #047857; }
    .provider-card-main { min-width: 0; }
    .provider-card-title {
      display: block;
      font-size: 14px;
      font-weight: 760;
      line-height: 1.3;
    }
    .provider-card-desc {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .provider-state {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #cbd5e1;
    }
    .provider-card.active .provider-state { background: #2563eb; }
    .api-config-main {
      padding: 28px 24px;
      overflow: auto;
    }
    .api-provider-summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 26px;
    }
    .provider-summary-left {
      display: flex;
      align-items: center;
      gap: 14px;
      min-width: 0;
    }
    .provider-summary-left .provider-icon {
      width: 42px;
      height: 42px;
      font-size: 18px;
    }
    .provider-summary-title { margin: 0; font-size: 22px; line-height: 1.2; }
    .provider-summary-subtitle { margin: 4px 0 0; color: var(--muted); font-size: 13px; }
    .switch {
      position: relative;
      width: 46px;
      height: 26px;
      flex: 0 0 auto;
    }
    .switch input {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
    }
    .switch span {
      position: absolute;
      inset: 0;
      border-radius: 999px;
      background: #cbd5e1;
      transition: background .16s ease;
    }
    .switch span::after {
      content: "";
      position: absolute;
      width: 20px;
      height: 20px;
      left: 3px;
      top: 3px;
      border-radius: 999px;
      background: #fff;
      box-shadow: 0 1px 4px rgba(15, 23, 42, .22);
      transition: transform .16s ease;
    }
    .switch input:checked + span { background: #2563eb; }
    .switch input:checked + span::after { transform: translateX(20px); }
    .api-config-actions {
      padding-top: 16px;
      margin-top: 18px;
      border-top: 1px solid var(--line);
      justify-content: space-between;
    }
    .api-field-hidden { display: none; }
    .api-advanced-fields { display: none; }
    .api-form-status { min-width: 0; flex: 1 1 auto; }
    .ollama-context-panel {
      display: none;
      margin: 14px 0 4px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }
    .ollama-context-panel.show { display: block; }
    .context-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }
    .context-row label { margin: 0; }
    .context-value {
      font-size: 13px;
      font-weight: 760;
      color: #2563eb;
    }
    .context-help {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      margin-bottom: 12px;
    }
    .context-slider {
      width: 100%;
      height: 8px;
      border: 0;
      padding: 0;
      accent-color: #2563eb;
      cursor: pointer;
    }
    .context-ticks {
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
    }
    .connection-result {
      display: none;
      margin: 14px 0 20px;
      padding: 14px 16px;
      border-radius: 8px;
      border: 1px solid transparent;
      font-size: 13px;
      line-height: 1.7;
    }
    .connection-result.show { display: block; }
    .connection-result.ok {
      background: #ecfdf3;
      border-color: #bbf7d0;
      color: #087443;
    }
    .connection-result.error {
      background: #fff1f2;
      border-color: #fecdd3;
      color: #b42318;
    }
    .connection-title {
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 780;
      margin-bottom: 2px;
    }
    .connection-mark {
      width: 18px;
      height: 18px;
      display: inline-grid;
      place-items: center;
      font-size: 14px;
    }
    .model-choice-block { margin-top: 18px; }
    .model-choice-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }
    .model-choice-head label { margin: 0; }
    .model-choice-list {
      display: block;
      max-height: 248px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .model-choice {
      min-height: 44px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid #eef2f7;
      background: #fff;
      cursor: pointer;
    }
    .model-choice:last-child { border-bottom: 0; }
    .model-choice:hover { background: #f8fbff; }
    .model-choice.selected {
      background: #eff6ff;
    }
    .model-choice-name {
      flex: 1 1 auto;
      min-width: 0;
      color: #172033;
      font-size: 13px;
      font-weight: 680;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .model-choice-radio {
      width: 18px;
      height: 18px;
      flex: 0 0 auto;
      accent-color: #2563eb;
    }
    .model-current-badge {
      display: none;
      flex: 0 0 auto;
      height: 24px;
      align-items: center;
      padding: 0 8px;
      border-radius: 6px;
      background: #dbeafe;
      color: #2563eb;
      font-size: 12px;
      font-weight: 720;
    }
    .model-choice.selected .model-current-badge { display: inline-flex; }
    .model-empty {
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      padding: 14px;
      color: var(--muted);
      font-size: 13px;
      background: #fbfdff;
    }
    .provider-select-hidden {
      position: absolute;
      width: 1px;
      height: 1px;
      opacity: 0;
      pointer-events: none;
    }
    @media (max-width: 1060px) {
      .shell { grid-template-columns: 1fr; }
      .app-header { align-items: flex-start; flex-direction: column; }
      .top-pills { justify-content: flex-start; }
      .api-config-layout { grid-template-columns: 1fr; max-height: calc(100vh - 130px); }
      .api-provider-pane { border-right: 0; border-bottom: 1px solid var(--line); }
      .provider-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .shell { padding: 14px; }
      .app-header { padding: 18px; }
      .grid, .triple-grid, .toggle-grid, .metric-strip, .workspace-tabs, .compare-hero, .compare-cards, .compare-controls, .secret-grid { grid-template-columns: 1fr; }
      .panel-pad, .section-head, .monitor-head, .run-body { padding-left: 16px; padding-right: 16px; }
      .modal-shell { padding: 12px; }
      .modal-panel { width: calc(100vw - 24px); }
      .provider-list { grid-template-columns: 1fr; }
      .api-provider-summary { align-items: flex-start; }
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
      <span class="pill"><span class="dot"></span>MinerU 固定 OCR</span>
      <span class="pill"><span class="dot"></span>可配置对话模型</span>
      <span class="pill"><span class="dot"></span>全量数据集测评</span>
      <button id="api-settings-button" class="secondary settings-button" type="button">API 设置</button>
    </div>
  </header>
  <main class="shell">
    <section class="workspace">
      <div class="workspace-tabs" role="tablist" aria-label="RAGEval workflows">
        <button class="workspace-tab active" type="button" data-workspace-tab="build">生成数据集</button>
        <button class="workspace-tab" type="button" data-workspace-tab="eval">DeepLocals 测评</button>
        <button class="workspace-tab" type="button" data-workspace-tab="compare">平台对比</button>
      </div>
      <div class="workflow-panel active" data-workspace-panel="build">
      <div class="section-head">
        <div class="kicker">Dataset Builder</div>
        <h2>生成问答对与证据</h2>
        <p>上传 PDF 或文档后，系统会解析全文、按证据块生成问题、答案和可核验证据，并输出到 E:\RAG\data。</p>
      </div>
      <form id="job-form" class="panel-pad">
        <label class="file-drop">
          <span class="file-title">上传原始文档</span>
          <span class="file-subtitle">支持 PDF 和 DeepLocals 同类上传文档；PDF 超过 200 页会自动分页送 MinerU 后再合并。</span>
          <input name="files" type="file" multiple required />
        </label>
        <div class="grid" style="margin-top:14px">
          <div class="field">
            <label>输出数据集名</label>
            <input id="build-dataset-name" name="dataset_name" value="custom_rag_eval" pattern="[A-Za-z0-9_.-]+" />
          </div>
          <div class="field">
            <label>解析服务</label>
            <input value="MinerU / 本地固定 Token" disabled />
          </div>
        </div>
        <div class="grid">
          <div class="field">
            <label>MinerU 模型</label>
            <select name="mineru_model">
              <option value="vlm">vlm</option>
              <option value="pipeline">pipeline</option>
            </select>
          </div>
          <div class="field">
            <label>语言</label>
            <input name="language" value="ch" />
          </div>
        </div>
        <div class="toggle-grid">
          <label class="checkrow"><input name="use_mineru" type="checkbox" checked />使用 MinerU 解析</label>
          <label class="checkrow"><input name="is_ocr" type="checkbox" checked />启用 OCR</label>
          <label class="checkrow"><input name="enable_table" type="checkbox" checked />表格识别</label>
          <label class="checkrow"><input name="enable_formula" type="checkbox" checked />公式识别</label>
        </div>
        <div class="block-title">
          <h2>问答生成策略</h2>
          <span class="pill">按块字数自适应</span>
        </div>
        <div class="metric-strip">
          <div class="metric"><span>LLM 后端</span><strong id="provider-metric">加载中</strong></div>
          <div class="metric"><span>模型</span><strong id="model-metric">加载中</strong></div>
          <div class="metric"><span>负例</span><strong>不生成</strong></div>
        </div>
        <div class="triple-grid">
          <div class="field">
            <label>目标问题数</label>
            <input name="target_count" type="number" min="1" max="5000" value="100" />
          </div>
          <div class="field">
            <label>每文件上限</label>
            <input name="max_per_source" type="number" min="1" max="5000" value="100" />
          </div>
          <div class="field">
            <label>分配方式</label>
            <div class="strategy">系统按标题块字数自动分配</div>
          </div>
        </div>
        <div class="action-row">
          <button id="submit-btn" type="submit">开始生成数据集</button>
          <span class="muted">生成完成后会自动刷新下方可测数据集列表。</span>
        </div>
      </form>
      </div>

      <div class="workflow-panel" data-workspace-panel="eval">
      <div class="section-head">
        <div class="kicker">Real RAG Evaluation</div>
        <h2>选择整个数据集进行真实测评</h2>
        <p>这里不再按单条 ID 测试。选中一个数据集后，会上传它对应的 corpus 到新的 DeepLocals 知识库，并用该数据集全部问答对完成评估。</p>
      </div>
      <form id="eval-form" class="panel-pad">
        <div class="field">
          <label>测评数据集</label>
          <input id="eval-dataset-value" name="dataset_name" type="hidden" />
          <div class="dataset-picker">
            <button id="dataset-picker-button" class="picker-button secondary" type="button" aria-expanded="false">
              <span>
                <span id="picker-title" class="picker-title">加载数据集...</span>
                <span id="picker-meta" class="picker-meta">正在扫描 E:\RAG\data</span>
              </span>
              <span class="chevron">v</span>
            </button>
            <div id="dataset-picker-menu" class="picker-menu">
              <input id="dataset-search" class="picker-search" type="search" placeholder="搜索数据集名" />
              <div id="dataset-list" class="dataset-list"></div>
            </div>
          </div>
        </div>
        <div class="grid">
          <div class="field">
            <label>DeepLocals 地址</label>
            <input name="api_base" value="http://127.0.0.1:3335" />
          </div>
          <div class="field">
            <label>知识库策略</label>
            <div class="strategy">每次自动新建隔离知识库</div>
          </div>
          <div class="field">
            <label>测评模式</label>
            <select name="mode">
              <option value="qa">检索 + 问答</option>
              <option value="retrieval">只测检索</option>
            </select>
          </div>
        </div>
        <div class="action-row">
          <button id="eval-btn" type="submit">运行真实测评</button>
          <span class="muted">会使用所选数据集的全部问答对，全量执行。</span>
        </div>
      </form>
      </div>

      <div class="workflow-panel" data-workspace-panel="compare">
      <div class="section-head">
        <div class="kicker">Platform Comparison</div>
        <h2>DeepLocals vs Cherry Studio</h2>
        <p>独立的端到端平台对比入口。不会影响上面的原版真实测评；两边会并发执行，DeepLocals 使用当前 API 设置里的对话模型，Cherry 使用下方填写的模型。</p>
      </div>
      <form id="compare-form" class="panel-pad compare-console">
        <div class="compare-hero">
          <div>
            <div class="kicker">Ready To Run</div>
            <h2>并发端到端平台对比</h2>
            <p>固定项使用当前预设；你只需要确认数据集、题量、检索块数和 API Key。</p>
          </div>
          <button id="compare-btn" class="compare-run-button" type="submit">开始对比</button>
        </div>
        <div class="compare-cards">
          <div class="compare-card">
            <span>DeepLocals</span>
            <strong>deepseek-ai/DeepSeek-V4-Flash</strong>
            <small>SiliconFlow · http://127.0.0.1:3335</small>
          </div>
          <div class="compare-card">
            <span>Cherry Studio</span>
            <strong>silicon:deepseek-ai/DeepSeek-V4-Flash</strong>
            <small>Knowledge Base: 6EmxXc46Hr7GZjYKU7bQU · Top 20</small>
          </div>
        </div>
        <div class="grid">
          <div class="field">
            <label>DeepLocals 地址</label>
            <input name="deepseekmine_api_base" value="http://127.0.0.1:3335" />
          </div>
          <div class="field">
            <label>DeepLocals 模型服务</label>
            <input name="deeplocal_provider" value="siliconflow" />
          </div>
          <div class="field">
            <label>DeepLocals 回答模型</label>
            <input name="deeplocal_model" value="deepseek-ai/DeepSeek-V4-Flash" />
          </div>
          <div class="field">
            <label>DeepLocals API 地址</label>
            <input name="deeplocal_api_url" value="https://api.siliconflow.cn/v1" />
          </div>
          <div class="field">
            <label>Cherry API 地址</label>
            <input name="cherry_api_base" value="http://127.0.0.1:23333" />
          </div>
          <div class="field">
            <label>Cherry Knowledge Base ID</label>
            <input name="cherry_knowledge_base_id" value="6EmxXc46Hr7GZjYKU7bQU" />
          </div>
          <div class="field">
            <label>Cherry 回答模型</label>
            <input name="cherry_model" value="silicon:deepseek-ai/DeepSeek-V4-Flash" />
          </div>
          <div class="field">
            <label>测评数据集</label>
            <select id="compare-dataset-select"></select>
          </div>
          <div class="field">
            <label>Cherry 检索块数</label>
            <input name="document_count" type="number" min="1" max="100" value="20" />
          </div>
          <div class="field">
            <label>题目上限</label>
            <input name="limit" type="number" min="0" max="5000" value="0" />
          </div>
        </div>
        <div class="field">
          <label>DeepLocals API Key</label>
          <input name="deeplocal_api_key" type="password" autocomplete="off" placeholder="留空则使用当前 API 设置或 SILICONFLOW_API_KEY" />
        </div>
        <div class="field">
          <label>Cherry API Key</label>
          <input name="cherry_api_key" type="password" autocomplete="off" placeholder="留空则使用服务进程环境变量 CHERRY_API_KEY" />
        </div>
        <div class="action-row">
          <button type="submit">并发平台对比</button>
          <span class="muted">题目上限 0 表示全量；评分规则和二次模型裁判与 DeepLocals 测评共用同一套逻辑。</span>
        </div>
      </form>
      </div>
    </section>

    <section class="monitor">
      <div class="monitor-head">
        <div>
          <div class="kicker">Run Monitor</div>
          <h2>任务状态</h2>
          <div id="job-id" class="job-label">尚未开始</div>
        </div>
        <div class="monitor-actions">
          <button id="cancel-btn" class="secondary danger-button" type="button" disabled>取消任务</button>
          <span id="badge" class="badge">idle</span>
        </div>
      </div>
      <div class="run-body">
        <div class="progress-row">
          <span>当前进度</span>
          <span id="progress-text">0%</span>
        </div>
        <progress id="progress" value="0" max="100"></progress>
        <div class="links" id="links"></div>
        <pre id="log">等待任务提交...</pre>
      </div>
    </section>
  </main>
  <div id="api-settings-modal" class="modal-shell" aria-hidden="true">
    <div class="modal-panel" role="dialog" aria-modal="true" aria-labelledby="api-settings-title">
      <div class="modal-head">
        <div>
          <div class="kicker">API Settings</div>
          <h2 id="api-settings-title">对话模型 API 配置</h2>
          <p>生成 QA、真实测评回答和裁判都会使用这里选择的对话模型。配置保存在本地 .rageval_api_config.json。</p>
        </div>
        <button id="api-settings-close" class="secondary modal-close" type="button" aria-label="关闭 API 设置">×</button>
      </div>
      <form id="llm-config-form" class="api-config-layout">
        <aside class="api-provider-pane">
          <div class="api-pane-head">
            <strong>选择提供商</strong>
            <button id="custom-provider-button" class="secondary tiny-button" type="button">+ 自定义</button>
          </div>
          <select id="llm-provider" class="provider-select-hidden" name="provider_id" aria-label="服务商"></select>
          <div id="provider-list" class="provider-list"></div>
        </aside>
        <section class="api-config-main">
          <div class="api-provider-summary">
            <div class="provider-summary-left">
              <span id="provider-summary-icon" class="provider-icon">AI</span>
              <div>
                <h3 id="provider-summary-title" class="provider-summary-title">云端大模型</h3>
                <p id="provider-summary-subtitle" class="provider-summary-subtitle">OpenAI-compatible 对话模型接口</p>
              </div>
            </div>
            <label class="switch" title="启用当前提供商">
              <input id="llm-enabled" name="enabled" type="checkbox" checked />
              <span></span>
            </label>
          </div>

          <div id="api-key-field" class="field">
            <label>API 密钥 *</label>
            <input id="llm-api-key" name="api_key" type="password" autocomplete="off" placeholder="输入当前提供商 API 密钥；MiMo 留空仍可读 .mimo_api_key" />
          </div>
          <div class="field">
            <label>API 地址</label>
            <input id="llm-api-url" name="api_url" value="https://api.siliconflow.cn/v1" />
          </div>
          <div id="ollama-context-panel" class="ollama-context-panel">
            <div class="context-row">
              <label>上下文长度</label>
              <span id="ollama-context-value" class="context-value">8K</span>
            </div>
            <div class="context-help">控制本地模型可使用的上下文窗口，数值越大占用显存或内存越多。</div>
            <input id="ollama-context-slider" class="context-slider" type="range" min="0" max="6" step="1" value="1" />
            <div class="context-ticks">
              <span>4K</span><span>8K</span><span>16K</span><span>32K</span><span>64K</span><span>128K</span><span>256K</span>
            </div>
          </div>
          <div id="llm-connection-status" class="connection-result" aria-live="polite">
            <div class="connection-title">
              <span id="llm-connection-mark" class="connection-mark">✓</span>
              <span id="llm-connection-title">连接成功</span>
            </div>
            <div id="llm-connection-detail">发现可用模型。</div>
          </div>
          <input id="llm-model" name="model" type="hidden" value="Qwen/Qwen3.5-4B" />
          <div class="model-choice-block">
            <div class="model-choice-head">
              <label>模型列表</label>
              <span id="model-choice-hint" class="muted">测试连接后可刷新列表</span>
            </div>
            <div id="model-choice-list" class="model-choice-list"></div>
          </div>
          <div class="grid api-advanced-fields" aria-hidden="true">
            <div class="field">
              <label>鉴权方式</label>
              <select id="llm-auth-type" name="auth_type">
                <option value="bearer">Authorization: Bearer</option>
                <option value="api-key">api-key</option>
              </select>
            </div>
            <div class="field">
              <label>最大输出 Tokens</label>
              <input id="llm-max-tokens" name="max_tokens" type="number" min="1" max="32768" value="4096" />
            </div>
          </div>
          <div class="grid api-advanced-fields" aria-hidden="true">
            <div class="field">
              <label>Temperature</label>
              <input id="llm-temperature" name="temperature" type="number" min="0" max="2" step="0.1" value="0" />
            </div>
            <div class="field">
              <label>Top P</label>
              <input id="llm-top-p" name="top_p" type="number" min="0.1" max="1" step="0.05" value="0.95" />
            </div>
          </div>
          <label class="checkrow api-advanced-fields" aria-hidden="true"><input id="llm-enable-thinking" name="enable_thinking" type="checkbox" />启用 SiliconFlow Qwen thinking</label>
          <div class="action-row api-config-actions">
            <button id="llm-test-btn" class="secondary" type="button">测试连接</button>
            <span id="llm-config-status" class="muted api-form-status">正在加载当前配置...</span>
            <button id="llm-save-btn" type="submit">保存配置</button>
          </div>
        </section>
      </form>
    </div>
  </div>
  <script>
    const form = document.getElementById('job-form');
    const evalForm = document.getElementById('eval-form');
    const compareForm = document.getElementById('compare-form');
    const workspaceTabs = Array.from(document.querySelectorAll('[data-workspace-tab]'));
    const workflowPanels = Array.from(document.querySelectorAll('[data-workspace-panel]'));
    const llmConfigForm = document.getElementById('llm-config-form');
    const apiSettingsButton = document.getElementById('api-settings-button');
    const apiSettingsModal = document.getElementById('api-settings-modal');
    const apiSettingsClose = document.getElementById('api-settings-close');
    const llmProvider = document.getElementById('llm-provider');
    const providerList = document.getElementById('provider-list');
    const providerSummaryIcon = document.getElementById('provider-summary-icon');
    const providerSummaryTitle = document.getElementById('provider-summary-title');
    const providerSummarySubtitle = document.getElementById('provider-summary-subtitle');
    const customProviderButton = document.getElementById('custom-provider-button');
    const llmEnabled = document.getElementById('llm-enabled');
    const apiKeyField = document.getElementById('api-key-field');
    const llmModel = document.getElementById('llm-model');
    const modelChoiceList = document.getElementById('model-choice-list');
    const modelChoiceHint = document.getElementById('model-choice-hint');
    const llmConnectionStatus = document.getElementById('llm-connection-status');
    const llmConnectionMark = document.getElementById('llm-connection-mark');
    const llmConnectionTitle = document.getElementById('llm-connection-title');
    const llmConnectionDetail = document.getElementById('llm-connection-detail');
    const ollamaContextPanel = document.getElementById('ollama-context-panel');
    const ollamaContextSlider = document.getElementById('ollama-context-slider');
    const ollamaContextValue = document.getElementById('ollama-context-value');
    const llmApiUrl = document.getElementById('llm-api-url');
    const llmApiKey = document.getElementById('llm-api-key');
    const llmAuthType = document.getElementById('llm-auth-type');
    const llmMaxTokens = document.getElementById('llm-max-tokens');
    const llmTemperature = document.getElementById('llm-temperature');
    const llmTopP = document.getElementById('llm-top-p');
    const llmEnableThinking = document.getElementById('llm-enable-thinking');
    const llmConfigStatus = document.getElementById('llm-config-status');
    const llmTestButton = document.getElementById('llm-test-btn');
    const providerMetric = document.getElementById('provider-metric');
    const modelMetric = document.getElementById('model-metric');
    const buildDatasetName = document.getElementById('build-dataset-name');
    const datasetValue = document.getElementById('eval-dataset-value');
    const pickerButton = document.getElementById('dataset-picker-button');
    const pickerMenu = document.getElementById('dataset-picker-menu');
    const pickerTitle = document.getElementById('picker-title');
    const pickerMeta = document.getElementById('picker-meta');
    const datasetSearch = document.getElementById('dataset-search');
    const datasetList = document.getElementById('dataset-list');
    const compareDatasetSelect = document.getElementById('compare-dataset-select');
    const button = document.getElementById('submit-btn');
    const evalButton = document.getElementById('eval-btn');
    const compareButton = document.getElementById('compare-btn');
    const cancelButton = document.getElementById('cancel-btn');
    const badge = document.getElementById('badge');
    const log = document.getElementById('log');
    const progress = document.getElementById('progress');
    const progressText = document.getElementById('progress-text');
    const links = document.getElementById('links');
    const jobIdEl = document.getElementById('job-id');
    let pollTimer = null;
    let datasets = [];
    let providerPresets = [];
    let availableModels = [];
    let running = false;
    let currentJobId = '';
    let savedLlmConfig = null;
    let ollamaContextLength = 8192;
    const providerKeyDrafts = {};
    const savedKeyPlaceholder = '__RAGEVAL_SAVED_API_KEY__';
    const ollamaContextOptions = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

    function syncButtons() {
      button.disabled = running;
      evalButton.disabled = running || !datasetValue.value;
      if (compareButton) compareButton.disabled = running || !datasetValue.value;
      if (cancelButton) cancelButton.disabled = !running || !currentJobId;
    }
    function setWorkspaceTab(name) {
      const target = name || 'build';
      workspaceTabs.forEach((tabButton) => {
        tabButton.classList.toggle('active', tabButton.dataset.workspaceTab === target);
      });
      workflowPanels.forEach((panel) => {
        panel.classList.toggle('active', panel.dataset.workspacePanel === target);
      });
      localStorage.setItem('rageval:workspaceTab', target);
      syncButtons();
    }
    function closePicker() {
      pickerMenu.classList.remove('open');
      pickerButton.setAttribute('aria-expanded', 'false');
    }
    function openApiSettings() {
      apiSettingsModal.classList.add('open');
      apiSettingsModal.setAttribute('aria-hidden', 'false');
      setTimeout(() => llmApiKey.focus(), 0);
    }
    function closeApiSettings() {
      apiSettingsModal.classList.remove('open');
      apiSettingsModal.setAttribute('aria-hidden', 'true');
      apiSettingsButton.focus();
    }
    function formatDatasetMeta(item) {
      const rows = `${item.rows || 0} 题`;
      const status = item.status_text || (item.has_corpus ? 'corpus 已就绪' : '旧格式，缺 corpus');
      return `${rows} · ${status}`;
    }
    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }
    function setSelectedDataset(item) {
      if (!item || !item.has_corpus) {
        datasetValue.value = '';
        if (compareDatasetSelect) compareDatasetSelect.value = '';
        pickerTitle.textContent = '暂无可测数据集';
        pickerMeta.textContent = '只有带 corpus 的数据集才能直接跑真实 RAG 测评';
        syncButtons();
        return;
      }
      datasetValue.value = item.name;
      localStorage.setItem('rageval:selectedDataset', item.name);
      pickerTitle.textContent = item.name;
      pickerMeta.textContent = formatDatasetMeta(item);
      if (compareDatasetSelect) compareDatasetSelect.value = item.name;
      syncButtons();
    }
    function selectedPreset(providerId) {
      return providerPresets.find((item) => item.id === providerId);
    }
    function contextLengthIndex(value) {
      const index = ollamaContextOptions.indexOf(Number(value));
      return index >= 0 ? index : 1;
    }
    function contextLengthLabel(value) {
      return `${Math.round(Number(value || 8192) / 1024)}K`;
    }
    function renderOllamaContextLength(value) {
      ollamaContextLength = Number(value || 8192);
      ollamaContextSlider.value = String(contextLengthIndex(ollamaContextLength));
      ollamaContextValue.textContent = contextLengthLabel(ollamaContextLength);
      const percent = (Number(ollamaContextSlider.value) / 6) * 100;
      ollamaContextSlider.style.background = `linear-gradient(to right, #2563eb 0%, #2563eb ${percent}%, #e5e7eb ${percent}%, #e5e7eb 100%)`;
    }
    async function loadOllamaContextLength() {
      try {
        const res = await fetch(`/api/ollama/context-config?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        renderOllamaContextLength(payload.context_length || 8192);
      } catch {
        renderOllamaContextLength(8192);
      }
    }
    async function saveOllamaContextLength() {
      try {
        const res = await fetch('/api/ollama/context-config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ context_length: ollamaContextLength }),
        });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok || payload.success === false) throw new Error(payload.error || '保存上下文长度失败');
        llmConfigStatus.textContent = `Ollama 上下文长度已保存：${contextLengthLabel(ollamaContextLength)}`;
      } catch (error) {
        llmConfigStatus.textContent = `上下文长度保存失败：${String(error && error.message ? error.message : error)}`;
      }
    }
    function defaultKeyPlaceholder() {
      if (llmProvider.value === 'ollama') return 'Ollama 本地无需密钥';
      return '输入当前提供商 API 密钥；MiMo 留空仍可读 .mimo_api_key';
    }
    function rememberProviderKey(providerId = llmProvider.value) {
      if (!providerId) return;
      const value = llmApiKey.value || '';
      if (value && value !== savedKeyPlaceholder) {
        providerKeyDrafts[providerId] = value;
      }
    }
    function showKeyForProvider(providerId) {
      if (providerId === 'ollama') {
        llmApiKey.value = '';
        llmApiKey.placeholder = '';
        return;
      }
      const draft = providerKeyDrafts[providerId];
      if (draft) {
        llmApiKey.value = draft;
        llmApiKey.placeholder = defaultKeyPlaceholder();
        return;
      }
      if (savedLlmConfig && savedLlmConfig.provider_id === providerId && savedLlmConfig.api_key_masked) {
        llmApiKey.value = savedKeyPlaceholder;
        llmApiKey.placeholder = `已保存 key：${savedLlmConfig.api_key_masked}`;
        return;
      }
      llmApiKey.value = '';
      llmApiKey.placeholder = defaultKeyPlaceholder();
    }
    function setConnectionStatus(type, title = '', detail = '') {
      llmConnectionStatus.className = 'connection-result';
      if (!type) {
        llmConnectionStatus.classList.remove('show', 'ok', 'error');
        return;
      }
      llmConnectionStatus.classList.add('show', type);
      llmConnectionMark.textContent = type === 'ok' ? '✓' : '!';
      llmConnectionTitle.textContent = title;
      llmConnectionDetail.textContent = detail;
    }
    function providerVisual(providerId) {
      const visuals = {
        siliconflow: { icon: 'SF', className: 'siliconflow' },
        deepseek: { icon: 'DS', className: 'deepseek' },
        qwen: { icon: 'Q', className: 'qwen' },
        mimo: { icon: 'MI', className: 'mimo' },
        ollama: { icon: 'OL', className: 'ollama' },
        custom: { icon: '+', className: 'custom' },
      };
      return visuals[providerId] || { icon: 'AI', className: 'custom' };
    }
    function shortProviderDescription(preset) {
      const descriptions = {
        siliconflow: '高性能 AI 推理服务平台',
        deepseek: 'DeepSeek AI 推理服务',
        qwen: '阿里云百炼智能大语言模型',
        mimo: '小米最新大模型，成本友好',
        ollama: '本地运行的 Ollama 模型',
        custom: '兼容 OpenAI Chat Completions',
      };
      return descriptions[preset.id] || preset.description || 'OpenAI-compatible 接口';
    }
    function updateProviderFormMode(providerId) {
      const isOllama = providerId === 'ollama';
      apiKeyField.classList.toggle('api-field-hidden', isOllama);
      ollamaContextPanel.classList.toggle('show', isOllama);
      llmApiKey.disabled = isOllama;
      llmTestButton.textContent = isOllama ? '刷新本地模型' : '测试连接';
      modelChoiceHint.textContent = isOllama ? '读取本地已下载模型' : modelChoiceHint.textContent;
    }
    function updateProviderSummary(providerId) {
      const preset = selectedPreset(providerId);
      if (!preset) return;
      const visual = providerVisual(providerId);
      providerSummaryIcon.textContent = visual.icon;
      providerSummaryIcon.className = `provider-icon ${visual.className}`;
      providerSummaryTitle.textContent = preset.name || providerId;
      providerSummarySubtitle.textContent = shortProviderDescription(preset);
    }
    function renderProviderList(activeId) {
      providerList.innerHTML = '';
      for (const preset of providerPresets) {
        const visual = providerVisual(preset.id);
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'provider-card' + (preset.id === activeId ? ' active' : '');
        card.innerHTML = `
          <span class="provider-icon ${visual.className}">${escapeHtml(visual.icon)}</span>
          <span class="provider-card-main">
            <span class="provider-card-title">${escapeHtml(preset.name || preset.id)}</span>
            <span class="provider-card-desc">${escapeHtml(shortProviderDescription(preset))}</span>
          </span>
          <span class="provider-state"></span>
        `;
        card.addEventListener('click', () => {
          rememberProviderKey();
          llmProvider.value = preset.id;
          applyProviderPreset(preset.id, false);
          renderProviderList(preset.id);
          updateProviderSummary(preset.id);
        });
        providerList.appendChild(card);
      }
    }
    function setModelChoices(models, selectedModel = '', sourceText = '默认模型') {
      const uniqueModels = [];
      for (const model of models || []) {
        const value = String(model || '').trim();
        if (value && !uniqueModels.includes(value)) uniqueModels.push(value);
      }
      if (selectedModel && !uniqueModels.includes(selectedModel)) uniqueModels.unshift(selectedModel);
      availableModels = uniqueModels;
      llmModel.value = selectedModel || uniqueModels[0] || '';
      modelChoiceList.innerHTML = '';
      if (!uniqueModels.length) {
        modelChoiceList.innerHTML = '<div class="model-empty">先点击“测试连接”获取模型列表。</div>';
        modelChoiceHint.textContent = '等待测试连接';
        return;
      }
      for (const model of uniqueModels) {
        const label = document.createElement('label');
        label.className = 'model-choice' + (model === llmModel.value ? ' selected' : '');
        label.innerHTML = `
          <input class="model-choice-radio" type="radio" name="model_choice" value="${escapeHtml(model)}" ${model === llmModel.value ? 'checked' : ''} />
          <span class="model-choice-name">${escapeHtml(model)}</span>
          <span class="model-current-badge">当前</span>
        `;
        label.querySelector('input').addEventListener('change', () => {
          llmModel.value = model;
          renderModelSelection();
        });
        modelChoiceList.appendChild(label);
      }
      modelChoiceHint.textContent = sourceText;
    }
    function renderModelSelection() {
      for (const card of modelChoiceList.querySelectorAll('.model-choice')) {
        const radio = card.querySelector('input');
        const selected = radio && radio.value === llmModel.value;
        card.classList.toggle('selected', Boolean(selected));
        if (radio) radio.checked = Boolean(selected);
      }
    }
    function applyProviderPreset(providerId, keepKey = true) {
      const preset = selectedPreset(providerId);
      if (!preset) return;
      llmApiUrl.value = preset.api_url || '';
      llmAuthType.value = preset.auth_type || 'bearer';
      llmMaxTokens.value = preset.max_tokens || 4096;
      llmTemperature.value = preset.temperature ?? 0;
      llmTopP.value = preset.top_p ?? 0.95;
      setModelChoices(preset.models || [], preset.default_model || '', '默认模型');
      llmEnableThinking.checked = Boolean((preset.extra_body || {}).enable_thinking);
      renderProviderList(providerId);
      updateProviderSummary(providerId);
      updateProviderFormMode(providerId);
      setConnectionStatus(null);
      if (!keepKey) showKeyForProvider(providerId);
      if (providerId === 'ollama') {
        setModelChoices([], '', '正在读取本地模型...');
        refreshProviderModels(true);
      }
    }
    function renderLlmConfig(config) {
      if (!providerPresets.length) return;
      savedLlmConfig = config;
      if (config.provider_id) delete providerKeyDrafts[config.provider_id];
      llmProvider.innerHTML = '';
      for (const preset of providerPresets) {
        const option = document.createElement('option');
        option.value = preset.id;
        option.textContent = preset.name;
        llmProvider.appendChild(option);
      }
      llmProvider.value = config.provider_id || 'mimo';
      llmApiUrl.value = config.api_url || '';
      llmAuthType.value = config.auth_type || 'bearer';
      llmMaxTokens.value = config.max_tokens || 4096;
      llmTemperature.value = config.temperature ?? 0;
      llmTopP.value = config.top_p ?? 0.95;
      llmEnableThinking.checked = Boolean((config.extra_body || {}).enable_thinking);
      llmEnabled.checked = config.enabled !== false;
      updateProviderFormMode(llmProvider.value);
      showKeyForProvider(llmProvider.value);
      setModelChoices(config.models || [], config.model || '', '已保存模型');
      renderProviderList(llmProvider.value);
      updateProviderSummary(llmProvider.value);
      const keyState = config.api_key_masked ? `已保存 key：${config.api_key_masked}` : '未保存 key；可使用环境变量或本地 key 文件';
      llmConfigStatus.textContent = `${config.provider_name || config.provider_id} / ${config.model || ''} / ${keyState}`;
      providerMetric.textContent = config.provider_name || config.provider_id || '未配置';
      modelMetric.textContent = config.model || '未配置';
    }
    async function loadLlmConfig() {
      try {
        const res = await fetch(`/api/llm-config?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        providerPresets = payload.presets || [];
        renderLlmConfig(payload.config || {});
      } catch (error) {
        llmConfigStatus.textContent = `配置加载失败：${String(error && error.message ? error.message : error)}`;
        providerMetric.textContent = '加载失败';
        modelMetric.textContent = '加载失败';
      }
    }
    function renderDatasetList(filterText = '') {
      const needle = filterText.trim().toLowerCase();
      const visible = datasets.filter((item) => item.name.toLowerCase().includes(needle));
      datasetList.innerHTML = '';
      if (!visible.length) {
        datasetList.innerHTML = '<div class="empty-state">没有匹配的数据集。</div>';
        return;
      }
      for (const item of visible) {
        const option = document.createElement('div');
        option.className = 'dataset-option' + (item.name === datasetValue.value ? ' selected' : '') + (!item.has_corpus ? ' unavailable' : '');
        const tagText = item.status_tag || (item.has_corpus ? '可测' : '旧格式');
        const tagWarn = item.status_class || (item.has_corpus ? '' : 'warn');
        option.innerHTML = `
          <button class="dataset-main" type="button" ${item.has_corpus ? '' : 'disabled'}>
            <span class="dataset-name">${escapeHtml(item.name)}</span>
            <span class="dataset-detail">${escapeHtml(formatDatasetMeta(item))}</span>
          </button>
          <span class="dataset-actions">
            <span class="tag ${tagWarn}">${escapeHtml(tagText)}</span>
            <button class="icon-danger" type="button" title="删除数据集" aria-label="删除 ${escapeHtml(item.name)}">×</button>
          </span>
        `;
        option.querySelector('.dataset-main').addEventListener('click', () => {
          if (!item.has_corpus) return;
          setSelectedDataset(item);
          renderDatasetList(datasetSearch.value);
          closePicker();
        });
        option.querySelector('.icon-danger').addEventListener('click', async (event) => {
          event.stopPropagation();
          await deleteDataset(item.name);
        });
        datasetList.appendChild(option);
      }
    }
    function renderCompareDatasetSelect() {
      if (!compareDatasetSelect) return;
      const available = datasets.filter((item) => item.has_corpus);
      compareDatasetSelect.innerHTML = '';
      if (!available.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = '暂无可测数据集';
        compareDatasetSelect.appendChild(option);
        compareDatasetSelect.disabled = true;
        return;
      }
      compareDatasetSelect.disabled = false;
      for (const item of available) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = `${item.name} · ${item.rows || 0} 题`;
        compareDatasetSelect.appendChild(option);
      }
      compareDatasetSelect.value = datasetValue.value || available[0].name;
    }
    async function deleteDataset(name) {
      if (running) return;
      if (!confirm(`删除数据集「${name}」？\\n会同时删除 JSON、corpus、报告和 assets 目录。`)) return;
      const res = await fetch(`/api/datasets/${encodeURIComponent(name)}`, { method: 'DELETE' });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        setBadge('error');
        log.textContent = payload.error || '删除失败';
        return;
      }
      if (datasetValue.value === name) {
        datasetValue.value = '';
        localStorage.removeItem('rageval:selectedDataset');
      }
      setBadge('idle');
      log.textContent = `已删除数据集：${name}`;
      await refreshDatasets();
    }
    async function refreshDatasets(preferredName) {
      pickerTitle.textContent = '加载数据集...';
      pickerMeta.textContent = '正在扫描 E:\\RAG\\data';
      try {
        const res = await fetch(`/api/datasets?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const payload = await res.json();
        datasets = payload.datasets || [];
      } catch (error) {
        datasets = [];
        datasetValue.value = '';
        pickerTitle.textContent = '数据集加载失败';
        pickerMeta.textContent = String(error && error.message ? error.message : error || '请稍后重试');
        datasetList.innerHTML = '<div class="empty-state">数据集列表加载失败，请刷新页面或稍后重试。</div>';
        syncButtons();
        return;
      }
      if (!datasets.length) {
        datasetValue.value = '';
        pickerTitle.textContent = '暂无数据集';
        pickerMeta.textContent = '先生成一个数据集，随后可在这里整集测评';
        datasetList.innerHTML = '<div class="empty-state">E:\\RAG\\data 里还没有可展示的数据集。</div>';
        syncButtons();
        return;
      }
      const wanted = preferredName || localStorage.getItem('rageval:selectedDataset') || buildDatasetName.value.trim();
      const available = datasets.filter((item) => item.has_corpus);
      const selected =
        available.find((item) => item.name === wanted) ||
        available.find((item) => item.name === datasetValue.value) ||
        available[0] ||
        null;
      setSelectedDataset(selected);
      renderCompareDatasetSelect();
      renderDatasetList(datasetSearch.value);
    }
    function setBadge(status) {
      badge.textContent = status || 'idle';
      badge.className = 'badge' + (status === 'done' ? ' done' : status === 'error' ? ' error' : '');
    }
    function buildLlmConfigPayload() {
      const data = new FormData(llmConfigForm);
      data.set('provider_name', (selectedPreset(llmProvider.value) || {}).name || llmProvider.value);
      data.set('models', availableModels.join('\n'));
      if (llmProvider.value === 'ollama') data.set('api_key', '');
      if (data.get('api_key') === savedKeyPlaceholder) data.set('api_key', '');
      data.set('enabled', llmEnabled.checked ? 'true' : '');
      const payload = Object.fromEntries(data.entries());
      payload.enable_thinking = llmEnableThinking.checked ? 'true' : 'false';
      return payload;
    }
    async function refreshProviderModels(auto = false) {
      llmTestButton.disabled = true;
      const isOllama = llmProvider.value === 'ollama';
      llmConfigStatus.textContent = isOllama ? '正在读取本地 Ollama 模型...' : '正在测试连接并获取模型列表...';
      try {
        const res = await fetch('/api/llm-config/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(buildLlmConfigPayload()),
        });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.error || (isOllama ? '读取本地模型失败' : '测试连接失败'));
        const selected = body.models && body.models.includes(llmModel.value) ? llmModel.value : (body.models || [])[0];
        setModelChoices(body.models || [], selected || '', isOllama ? '本地已下载模型' : '测试连接返回');
        setConnectionStatus('ok', isOllama ? '已读取本地模型' : '连接成功', body.message || `发现 ${(body.models || []).length} 个可用模型。`);
        llmConfigStatus.textContent = isOllama ? '请选择一个本地模型后保存。' : '请选择一个模型后保存。';
      } catch (error) {
        const message = String(error && error.message ? error.message : error);
        setConnectionStatus('error', isOllama ? '读取失败' : '连接失败', message);
        llmConfigStatus.textContent = isOllama ? '请确认 Ollama 已启动并已下载模型。' : '请检查 API 密钥和 API 地址。';
        if (isOllama && auto) {
          setModelChoices([], '', '未检测到本地模型');
        }
      } finally {
        llmTestButton.disabled = false;
      }
    }
    async function poll(jobId) {
      let job;
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        job = await res.json();
      } catch (error) {
        running = false;
        currentJobId = '';
        clearInterval(pollTimer);
        setBadge('idle');
        jobIdEl.textContent = '尚未开始';
        log.textContent = `任务状态已重置：旧任务 ${jobId} 不存在或服务已重启。`;
        syncButtons();
        return;
      }
      setBadge(job.status);
      progress.value = job.progress || 0;
      progressText.textContent = `${Math.round(job.progress || 0)}%`;
      log.textContent = (job.logs || []).join('\n') || '等待日志...';
      log.scrollTop = log.scrollHeight;
      links.innerHTML = '';
      if (job.output_url) links.innerHTML += `<a href="${job.output_url}">下载 JSONL</a>`;
      if (job.corpus_url) links.innerHTML += `<a href="${job.corpus_url}">下载 Corpus</a>`;
      if (job.report_url) links.innerHTML += `<a href="${job.report_url}">下载报告</a>`;
      if (job.summary_url) links.innerHTML += `<a href="${job.summary_url}">下载对比摘要</a>`;
      if (job.csv_url) links.innerHTML += `<a href="${job.csv_url}">下载 CSV</a>`;
      if (job.assets_manifest) links.innerHTML += `<a href="/download?path=${encodeURIComponent(job.assets_manifest)}">资产 Manifest</a>`;
      if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled' || job.status === 'idle') {
        running = false;
        currentJobId = '';
        clearInterval(pollTimer);
        if (job.status === 'done' && job.output_path) {
          await refreshDatasets(buildDatasetName.value.trim());
        }
        syncButtons();
      }
    }
    if (cancelButton) {
      cancelButton.addEventListener('click', async () => {
        if (!currentJobId) return;
        cancelButton.disabled = true;
        setBadge('cancelling');
        await fetch(`/api/jobs/${currentJobId}/cancel`, { method: 'POST' }).catch(() => null);
        poll(currentJobId);
      });
    }
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      running = true;
      syncButtons();
      closePicker();
      log.textContent = '';
      links.innerHTML = '';
      progress.value = 0;
      progressText.textContent = '0%';
      setBadge('queued');
      const data = new FormData(form);
      const res = await fetch('/api/jobs', { method: 'POST', body: data });
      const payload = await res.json();
      if (!res.ok) {
        setBadge('error');
        running = false;
        syncButtons();
        log.textContent = payload.error || '提交失败';
        return;
      }
      jobIdEl.textContent = payload.job_id;
      currentJobId = payload.job_id;
      syncButtons();
      pollTimer = setInterval(() => poll(payload.job_id), 1200);
      poll(payload.job_id);
    });
    buildDatasetName.addEventListener('input', () => {
      const wanted = buildDatasetName.value.trim();
      const item = datasets.find((dataset) => dataset.name === wanted && dataset.has_corpus);
      if (item) setSelectedDataset(item);
      renderDatasetList(datasetSearch.value);
    });
    pickerButton.addEventListener('click', () => {
      const open = pickerMenu.classList.toggle('open');
      pickerButton.setAttribute('aria-expanded', String(open));
      if (open) {
        datasetSearch.focus();
        datasetSearch.select();
      }
    });
    document.addEventListener('click', (event) => {
      if (!pickerMenu.contains(event.target) && !pickerButton.contains(event.target)) closePicker();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        closePicker();
        if (apiSettingsModal.classList.contains('open')) closeApiSettings();
      }
    });
    apiSettingsButton.addEventListener('click', openApiSettings);
    apiSettingsClose.addEventListener('click', closeApiSettings);
    apiSettingsModal.addEventListener('click', (event) => {
      if (event.target === apiSettingsModal) closeApiSettings();
    });
    customProviderButton.addEventListener('click', () => {
      rememberProviderKey();
      llmProvider.value = 'custom';
      applyProviderPreset('custom', false);
      llmApiUrl.focus();
      llmApiUrl.select();
    });
    llmApiKey.addEventListener('focus', () => {
      if (llmApiKey.value === savedKeyPlaceholder) llmApiKey.select();
    });
    ollamaContextSlider.addEventListener('input', () => {
      const index = Number(ollamaContextSlider.value);
      renderOllamaContextLength(ollamaContextOptions[index] || 8192);
    });
    ollamaContextSlider.addEventListener('change', saveOllamaContextLength);
    datasetSearch.addEventListener('input', () => renderDatasetList(datasetSearch.value));
    if (compareDatasetSelect) {
      compareDatasetSelect.addEventListener('change', () => {
        const item = datasets.find((dataset) => dataset.name === compareDatasetSelect.value && dataset.has_corpus);
        if (item) {
          setSelectedDataset(item);
          renderDatasetList(datasetSearch.value);
        }
      });
    }
    llmProvider.addEventListener('change', () => {
      rememberProviderKey();
      applyProviderPreset(llmProvider.value, false);
    });
    llmTestButton.addEventListener('click', () => refreshProviderModels(false));
    llmConfigForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const payload = buildLlmConfigPayload();
      const res = await fetch('/api/llm-config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        llmConfigStatus.textContent = body.error || '保存失败';
        return;
      }
      if (payload.provider_id === 'ollama') await saveOllamaContextLength();
      renderLlmConfig(body.config || {});
      llmConfigStatus.textContent = `已保存：${body.config.provider_name} / ${body.config.model}`;
    });
    evalForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!datasetValue.value) {
        setBadge('error');
        log.textContent = '请选择一个带 corpus 的数据集。旧数据集如果没有 corpus，需要先通过本 UI 重新生成。';
        return;
      }
      running = true;
      syncButtons();
      closePicker();
      log.textContent = '';
      links.innerHTML = '';
      progress.value = 0;
      progressText.textContent = '0%';
      setBadge('queued');
      const data = new FormData(evalForm);
      const res = await fetch('/api/eval-jobs', { method: 'POST', body: data });
      const payload = await res.json();
      if (!res.ok) {
        setBadge('error');
        running = false;
        syncButtons();
        log.textContent = payload.error || '提交失败';
        return;
      }
      jobIdEl.textContent = payload.job_id;
      currentJobId = payload.job_id;
      syncButtons();
      pollTimer = setInterval(() => poll(payload.job_id), 1200);
      poll(payload.job_id);
    });
    if (compareForm) {
      compareForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!datasetValue.value) {
          setBadge('error');
          log.textContent = '请选择一个带 corpus 的数据集后再启动平台对比。';
          return;
        }
        running = true;
        syncButtons();
        closePicker();
        log.textContent = '';
        links.innerHTML = '';
        progress.value = 0;
        progressText.textContent = '0%';
        setBadge('queued');
        const data = new FormData(compareForm);
        data.set('dataset_name', datasetValue.value);
        const res = await fetch('/api/compare-jobs', { method: 'POST', body: data });
        const payload = await res.json();
        if (!res.ok) {
          setBadge('error');
          running = false;
          syncButtons();
          log.textContent = payload.error || '提交失败';
          return;
        }
        jobIdEl.textContent = payload.job_id;
        currentJobId = payload.job_id;
        syncButtons();
        pollTimer = setInterval(() => poll(payload.job_id), 1200);
        poll(payload.job_id);
      });
    }
    workspaceTabs.forEach((tabButton) => {
      tabButton.addEventListener('click', () => setWorkspaceTab(tabButton.dataset.workspaceTab));
    });
    setWorkspaceTab(localStorage.getItem('rageval:workspaceTab') || 'compare');
    loadOllamaContextLength();
    loadLlmConfig();
    refreshDatasets(localStorage.getItem('rageval:selectedDataset') || buildDatasetName.value.trim());
  </script>
</body>
</html>
"""


class RAGDatasetUIHandler(BaseHTTPRequestHandler):
    server_version = "RAGDatasetUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            self.send_json(get_job(job_id) or {"error": "not found"}, HTTPStatus.OK if get_job(job_id) else HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/datasets":
            self.send_json(list_datasets())
            return
        if parsed.path == "/api/llm-config":
            self.send_json({"config": masked_config(), "presets": provider_presets_payload()})
            return
        if parsed.path == "/api/ollama/context-config":
            self.send_json(load_ollama_context_config())
            return
        if parsed.path == "/download":
            query = parse_qs(parsed.query)
            target = Path(query.get("path", [""])[0])
            if not is_download_allowed(target) or not target.exists():
                self.send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
                return
            self.send_file(target)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path.strip("/").split("/")[2]
            self.send_json(cancel_job(job_id))
            return
        if parsed.path == "/api/ollama/context-config":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw or "{}")
                context_length = payload.get("context_length")
                if not isinstance(context_length, int) or context_length < 4096 or context_length > 262144:
                    self.send_json({"error": "Invalid context_length. Must be a number between 4096 and 262144"}, HTTPStatus.BAD_REQUEST)
                    return
                config = save_ollama_context_config(context_length)
                self.send_json({"success": True, **config})
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/llm-config":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw or "{}")
                config = config_from_form(payload)
                preserve_saved_api_key(config)
                save_llm_config(config)
                self.send_json({"ok": True, "config": masked_config(config), "presets": provider_presets_payload()})
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/llm-config/test":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw or "{}")
                config = config_from_form(payload)
                preserve_saved_api_key(config)
                models = list_provider_models(config)
                self.send_json({
                    "ok": True,
                    "models": models,
                    "message": f"连接成功，获取到 {len(models)} 个模型。",
                })
            except Exception as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path not in {"/api/jobs", "/api/eval-jobs", "/api/compare-jobs"}:
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            if parsed.path == "/api/eval-jobs":
                job_id = create_eval_job(form)
            elif parsed.path == "/api/compare-jobs":
                job_id = create_compare_job(form)
            else:
                job_id = create_job(form)
            self.send_json({"job_id": job_id})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/datasets/"):
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        raw_name = parsed.path.rsplit("/", 1)[-1]
        try:
            from urllib.parse import unquote

            name = sanitize_dataset_name(unquote(raw_name))
            self.send_json(delete_dataset(name))
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[ui] {self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return json.loads(json.dumps(job, ensure_ascii=False)) if job else None


def update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)


def cancel_job(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {"error": "not found"}
        if job.get("status") in {"done", "error", "cancelled"}:
            return {"ok": True, "status": job.get("status")}
        job["cancel_requested"] = True
        job["status"] = "cancelling"
    log_job(job_id, "已请求取消，正在等待当前步骤安全停下")
    return {"ok": True, "status": "cancelling"}


def is_cancel_requested(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("cancel_requested"))


def check_cancelled(job_id: str) -> None:
    if is_cancel_requested(job_id):
        raise JobCancelled("任务已取消")


def log_job(job_id: str, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with JOBS_LOCK:
        JOBS[job_id]["logs"].append(f"[{stamp}] {message}")
        JOBS[job_id]["logs"] = JOBS[job_id]["logs"][-500:]


def field_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    item = form[name] if name in form else None
    if item is None or isinstance(item, list):
        return default
    return str(item.value or default)


def field_checked(form: cgi.FieldStorage, name: str) -> bool:
    return name in form


def int_field(form: cgi.FieldStorage, name: str, default: int) -> int:
    try:
        return int(field_value(form, name, str(default)))
    except ValueError:
        return default


def pct(value) -> float:
    try:
        return round(float(value) * 100, 2)
    except (TypeError, ValueError):
        return 0.0


def comparison_row(platform: str, report: dict) -> dict:
    summary = report.get("summary") or {}
    return {
        "platform": platform,
        "dataset": report.get("dataset_name") or report.get("dataset") or "",
        "qa_accuracy_percent": pct(summary.get("qa_accuracy")),
        "qa_correct": summary.get("qa_correct", 0),
        "qa_total": summary.get("qa_total", summary.get("total", 0)),
        "evidence_hit_rate_percent": pct(summary.get("evidence_hit_rate")),
        "evidence_recall_percent": pct(summary.get("evidence_recall")),
        "mrr": round(float(summary.get("mrr") or 0.0), 4),
        "avg_retrieved_count": round(float(summary.get("avg_retrieved_count") or 0.0), 2),
        "answer_model": report.get("answer_model") or report.get("model") or "",
        "report_path": report.get("output_path") or "",
        "completed": report.get("completed", False),
        "stopped_reason": report.get("stopped_reason") or "",
    }


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_dataset_corpus(dataset_name: str) -> Path | None:
    report_path = DATA_DIR / f"{dataset_name}.json.report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for key in ("corpus_path", "corpus", "corpus_source"):
                value = str(report.get(key) or "").strip()
                if value:
                    path = Path(value)
                    if path.exists():
                        return path
        except Exception:
            pass
    for suffix in (".corpus.txt", ".corpus.md"):
        path = DATA_DIR / f"{dataset_name}{suffix}"
        if path.exists():
            return path
    return None


def list_datasets() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    datasets: list[dict] = []
    for path in sorted(DATA_DIR.glob("*.json")):
        if path.name.endswith(".report.json"):
            continue
        name = path.stem
        corpus_path = find_dataset_corpus(name)
        asset_dir = ASSETS_DIR / name
        manifest_path = asset_dir / "manifest.json"
        has_assets = asset_dir.exists()
        has_mineru_assets = False
        if has_assets:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                has_mineru_assets = bool(manifest.get("mineru_markdown_files"))
            except Exception:
                has_mineru_assets = False
            if not has_mineru_assets:
                has_mineru_assets = any(
                    path.with_suffix(path.suffix + ".mineru.json").exists()
                    for path in (asset_dir / "mineru").glob("*.mineru.md")
                )
        rows = 0
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                rows = sum(1 for line in handle if line.strip())
        except Exception:
            rows = 0
        has_corpus = corpus_path is not None and corpus_path.exists()
        if has_corpus and has_mineru_assets:
            status_text = "DeepLocals 对齐可测"
            status_tag = "可测"
            status_class = ""
        elif has_corpus:
            status_text = "文本语料可测"
            status_tag = "可测"
            status_class = ""
        else:
            status_text = "旧格式，缺 corpus，需用本 UI 重新生成"
            status_tag = "旧格式"
            status_class = "warn"
        datasets.append(
            {
                "name": name,
                "path": str(path),
                "rows": rows,
                "has_corpus": has_corpus,
                "has_assets": has_assets,
                "has_mineru_assets": has_mineru_assets,
                "status_text": status_text,
                "status_tag": status_tag,
                "status_class": status_class,
                "corpus_path": str(corpus_path) if has_corpus and corpus_path else "",
                "updated_at": path.stat().st_mtime,
            }
        )
    datasets.sort(key=lambda item: item["updated_at"], reverse=True)
    return {"datasets": datasets}


def delete_dataset(dataset_name: str) -> dict:
    name = sanitize_dataset_name(dataset_name)
    if not name:
        raise ValueError("数据集名为空")
    deleted: list[str] = []
    candidates = [
        DATA_DIR / f"{name}.json",
        DATA_DIR / f"{name}.corpus.txt",
        DATA_DIR / f"{name}.corpus.md",
        DATA_DIR / f"{name}.corpus.generated.md",
        DATA_DIR / f"{name}.json.report.json",
    ]
    for path in candidates:
        if path.exists():
            path.unlink()
            deleted.append(str(path))
    asset_dir = ASSETS_DIR / name
    if asset_dir.exists():
        resolved_root = ASSETS_DIR.resolve()
        resolved_asset = asset_dir.resolve()
        if not str(resolved_asset).lower().startswith(str(resolved_root).lower()):
            raise RuntimeError(f"unsafe asset path: {asset_dir}")
        shutil.rmtree(asset_dir)
        deleted.append(str(asset_dir))
    for report in RESULT_DIR.glob(f"deepseekmine_*_{name}_*.json"):
        if report.exists() and is_download_allowed(report):
            report.unlink()
            deleted.append(str(report))
    if not deleted:
        raise FileNotFoundError(f"未找到数据集：{name}")
    return {"ok": True, "deleted": deleted}


def create_job(form: cgi.FieldStorage) -> str:
    file_items = form["files"] if "files" in form else []
    if not isinstance(file_items, list):
        file_items = [file_items]
    file_items = [item for item in file_items if getattr(item, "filename", "")]
    if not file_items:
        raise ValueError("请至少上传一个文件")

    dataset_name = sanitize_dataset_name(field_value(form, "dataset_name", "custom_rag_eval"))
    job_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / job_id
    upload_dir = run_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for item in file_items:
        filename = sanitize_filename(Path(item.filename).name)
        target = unique_path(upload_dir / filename)
        with target.open("wb") as f:
            shutil.copyfileobj(item.file, f)
        saved_paths.append(str(target))

    target_count = int_field(form, "target_count", 100)
    max_per_source = int_field(form, "max_per_source", target_count)
    max_per_source = max(1, min(max_per_source, max(target_count, 5000)))
    config = {
        "dataset_name": dataset_name,
        "use_mineru": field_checked(form, "use_mineru"),
        "mineru_model": field_value(form, "mineru_model", "vlm"),
        "language": field_value(form, "language", "ch"),
        "is_ocr": field_checked(form, "is_ocr"),
        "enable_table": field_checked(form, "enable_table"),
        "enable_formula": field_checked(form, "enable_formula"),
        "backend": "configured",
        "model": "",
        "base_url": "",
        "target_count": target_count,
        "max_per_source": max_per_source,
        "items_per_chunk": 1,
        "negative_count": 0,
        "allow_heuristic": False,
    }
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "logs": [],
            "created_at": time.time(),
            "run_dir": str(run_dir),
            "output_path": "",
            "report_path": "",
            "output_url": "",
            "report_url": "",
            "corpus_path": "",
            "corpus_url": "",
            "assets_dir": "",
            "assets_manifest": "",
            "cancel_requested": False,
        }
    thread = threading.Thread(target=run_job, args=(job_id, saved_paths, config), daemon=True)
    thread.start()
    return job_id


def create_eval_job(form: cgi.FieldStorage) -> str:
    dataset_name = sanitize_dataset_name(field_value(form, "dataset_name", "custom_rag_eval"))
    dataset_path = DATA_DIR / f"{dataset_name}.json"
    corpus_path = find_dataset_corpus(dataset_name)
    if not dataset_path.exists():
        raise ValueError(f"数据集不存在：{dataset_path}")
    if corpus_path is None or not corpus_path.exists():
        raise ValueError(f"corpus 不存在：{DATA_DIR / f'{dataset_name}.corpus.txt'} 或 {DATA_DIR / f'{dataset_name}.corpus.md'}。请先用本 UI 生成该数据集。")

    job_id = uuid.uuid4().hex[:12]
    knowledge_label = f"rag_eval_{dataset_name}_{job_id}_{int(time.time())}"
    config = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "corpus_path": str(corpus_path),
        "api_base": field_value(form, "api_base", "http://127.0.0.1:3335").rstrip("/"),
        "knowledge_label": knowledge_label,
        "mode": field_value(form, "mode", "qa"),
        "limit": 0,
    }
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "logs": [],
            "created_at": time.time(),
            "run_dir": str(RUNS_DIR / job_id),
            "output_path": "",
            "report_path": "",
            "output_url": "",
            "report_url": "",
            "corpus_path": "",
            "corpus_url": "",
            "cancel_requested": False,
        }
    thread = threading.Thread(target=run_eval_job, args=(job_id, config), daemon=True)
    thread.start()
    return job_id


def create_compare_job(form: cgi.FieldStorage) -> str:
    dataset_name = sanitize_dataset_name(field_value(form, "dataset_name", "custom_rag_eval"))
    dataset_path = DATA_DIR / f"{dataset_name}.json"
    corpus_path = find_dataset_corpus(dataset_name)
    if not dataset_path.exists():
        raise ValueError(f"数据集不存在：{dataset_path}")
    if corpus_path is None or not corpus_path.exists():
        raise ValueError(f"corpus 不存在：{DATA_DIR / f'{dataset_name}.corpus.txt'} 或 {DATA_DIR / f'{dataset_name}.corpus.md'}。请先生成该数据集。")

    job_id = uuid.uuid4().hex[:12]
    config = {
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "corpus_path": str(corpus_path),
        "deepseekmine_api_base": field_value(form, "deepseekmine_api_base", "http://127.0.0.1:3335").rstrip("/"),
        "deeplocal_provider": field_value(form, "deeplocal_provider", "siliconflow"),
        "deeplocal_model": field_value(form, "deeplocal_model", "deepseek-ai/DeepSeek-V4-Flash"),
        "deeplocal_api_url": field_value(form, "deeplocal_api_url", "https://api.siliconflow.cn/v1"),
        "deeplocal_api_key": field_value(form, "deeplocal_api_key", ""),
        "cherry_api_base": field_value(form, "cherry_api_base", "http://127.0.0.1:23333").rstrip("/"),
        "cherry_api_key": field_value(form, "cherry_api_key", ""),
        "cherry_knowledge_base_id": field_value(form, "cherry_knowledge_base_id", ""),
        "cherry_model": field_value(form, "cherry_model", "silicon:deepseek-ai/DeepSeek-V4-Flash"),
        "document_count": max(1, int_field(form, "document_count", 20)),
        "limit": max(0, int_field(form, "limit", 0)),
    }
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0,
            "logs": [],
            "created_at": time.time(),
            "run_dir": str(RUNS_DIR / job_id),
            "output_path": "",
            "report_path": "",
            "output_url": "",
            "report_url": "",
            "summary_path": "",
            "summary_url": "",
            "csv_path": "",
            "csv_url": "",
            "corpus_path": "",
            "corpus_url": "",
            "cancel_requested": False,
        }
    thread = threading.Thread(target=run_compare_eval_job, args=(job_id, config), daemon=True)
    thread.start()
    return job_id


def run_job(job_id: str, saved_paths: list[str], config: dict) -> None:
    try:
        update_job(job_id, status="running", progress=5)
        check_cancelled(job_id)
        run_dir = Path(get_job(job_id)["run_dir"])
        log_job(job_id, f"收到 {len(saved_paths)} 个文件")
        ensure_llm_config()

        input_paths = [Path(p) for p in saved_paths]
        asset_dir = prepare_dataset_asset_dir(config["dataset_name"])
        original_asset_paths = copy_files_to_dir(input_paths, asset_dir / "original")
        log_job(job_id, f"已保存原始文件：{asset_dir / 'original'}")
        file_hashes = {str(path): sha256_file(path) for path in input_paths}
        builder_inputs: list[str] = []
        mineru_report: list[dict] = []
        mineru_candidates = mineru_supported_paths(input_paths)
        local_candidates = [p for p in input_paths if p not in mineru_candidates]

        if config["use_mineru"] and mineru_candidates:
            update_job(job_id, progress=12)
            check_cancelled(job_id)
            uncached_mineru_candidates: list[Path] = []
            for candidate in mineru_candidates:
                cache_entry = load_mineru_cache_entry(file_hashes[str(candidate)])
                if cache_entry:
                    asset_markdown = copy_cached_mineru_to_assets(cache_entry, asset_dir / "mineru")
                    builder_inputs.append(str(asset_markdown))
                    mineru_report.append(
                        {
                            "original_path": str(candidate),
                            "success": True,
                            "cache_hit": True,
                            "file_sha256": file_hashes[str(candidate)],
                            "markdown_path": str(asset_markdown),
                            "metadata_path": str(asset_markdown.with_suffix(asset_markdown.suffix + ".meta.json")),
                            "combined_json_path": str(asset_markdown.with_suffix(asset_markdown.suffix + ".mineru.json")),
                        }
                    )
                    log_job(job_id, f"复用 MinerU 缓存：{candidate.name} -> {file_hashes[str(candidate)][:12]}")
                else:
                    uncached_mineru_candidates.append(candidate)

            log_job(job_id, f"MinerU 解析文件数：{len(uncached_mineru_candidates)}，缓存命中：{len(mineru_candidates) - len(uncached_mineru_candidates)}")
            if uncached_mineru_candidates:
                mineru_token = ensure_mineru_key()
                options = MineruOptions(
                    token=mineru_token,
                    model_version=config["mineru_model"],
                    language=config["language"],
                    is_ocr=config["is_ocr"],
                    enable_formula=config["enable_formula"],
                    enable_table=config["enable_table"],
                )
                mineru_dir = run_dir / "mineru_markdown"
                def mineru_progress(msg: str) -> None:
                    check_cancelled(job_id)
                    log_job(job_id, msg)

                results = process_with_mineru(
                    uncached_mineru_candidates,
                    mineru_dir,
                    options,
                    progress=mineru_progress,
                )
                mineru_report.extend([r.__dict__ for r in results])
                for result in results:
                    if result.success and result.markdown_path:
                        original = Path(result.original_path)
                        digest = file_hashes.get(str(original)) or sha256_file(original)
                        store_mineru_result_in_cache(result, digest, original)
                        asset_markdown = copy_mineru_result_to_assets(result, asset_dir / "mineru")
                        builder_inputs.append(str(asset_markdown))
                        log_job(job_id, f"写入 MinerU 缓存：{original.name} -> {digest[:12]}")
                    else:
                        log_job(job_id, f"MinerU 失败：{Path(result.original_path).name} - {result.error}")
        else:
            if mineru_candidates and not config["use_mineru"]:
                log_job(job_id, "已关闭 MinerU，使用本地解析器处理支持文件")
            builder_inputs.extend(str(p) for p in mineru_candidates)

        builder_inputs.extend(str(p) for p in local_candidates)
        if not builder_inputs:
            raise RuntimeError("没有可用于生成数据集的解析结果")
        check_cancelled(job_id)

        deepseekmine_chunk_sidecars: list[str] = []
        for markdown_input in [Path(p) for p in builder_inputs if Path(p).suffix.lower() in {".md", ".markdown"}]:
            check_cancelled(job_id)
            sidecar = write_deepseekmine_compatible_chunk_sidecar(markdown_input, config["dataset_name"])
            if sidecar:
                deepseekmine_chunk_sidecars.append(str(sidecar))
                log_job(job_id, f"复用 DeepLocals 分块：{markdown_input.name} -> {sidecar.name}")
        if builder_inputs and not deepseekmine_chunk_sidecars:
            log_job(job_id, "未生成 DeepLocals 分块 sidecar，将使用本地 MinerU 语义分块兜底")

        update_job(job_id, progress=55)
        check_cancelled(job_id)
        output_path = DATA_DIR / f"{config['dataset_name']}.json"
        corpus_path = DATA_DIR / f"{config['dataset_name']}.corpus.md"
        write_eval_corpus(corpus_path, [Path(p) for p in builder_inputs])
        log_job(job_id, f"开始生成 QA：输出 {output_path}")

        def qa_progress(event: str, **payload) -> None:
            check_cancelled(job_id)
            if event == "parse_doc":
                log_job(job_id, f"QA 输入：{Path(payload.get('source', '')).name}，{payload.get('chars', 0)} 字，解析器 {payload.get('parser', '')}")
                return
            if event == "parse_skip":
                log_job(job_id, f"QA 跳过：{Path(payload.get('source', '')).name} - {payload.get('reason', '')}")
                return
            if event == "chunk_plan":
                splitter = {
                    "mineru_semantic": "MinerU 语义 JSON + Markdown 标题层级",
                    "markdown_heading": "Markdown 标题层级",
                }.get(payload.get("splitter"), "段落窗口")
                markdown_count = payload.get("markdown_document_count", 0)
                semantic_count = payload.get("mineru_semantic_document_count", 0)
                log_job(job_id, f"QA 分块完成：{payload.get('document_count', 0)} 个文档，{payload.get('block_count', 0)} 个证据块，策略 {splitter}，Markdown 文档 {markdown_count} 个，语义 sidecar {semantic_count} 个")
                return
            if event == "qa_plan":
                log_job(job_id, f"QA 自适应分配：目标 {payload.get('target_count', 0)} 题，候选调用 {payload.get('planned_total', 0)} 次，覆盖 {payload.get('active_blocks', 0)}/{payload.get('block_count', 0)} 个证据块，单块最多 {payload.get('max_per_block', 0)} 题；若有拒绝会自动向后补块")
                return
            if event == "api_quota_exhausted":
                row_count = int(payload.get("row_count") or 0)
                target_count = max(1, int(payload.get("target_count") or config["target_count"]))
                update_job(job_id, progress=min(99, int(55 + min(1.0, row_count / target_count) * 40)))
                log_job(job_id, f"API 额度不足或请求超限，已保护性中止；已生成 {payload.get('row_count', 0)}/{payload.get('target_count', 0)} 题。原因：{payload.get('reason', '')}")
                return
            if event not in {"qa_chunk_start", "qa_chunk_done", "qa_chunk_skip"}:
                return

            block_index = int(payload.get("block_index") or 0)
            total_blocks = max(1, int(payload.get("total_blocks") or 1))
            row_count = int(payload.get("row_count") or 0)
            target_count = max(1, int(payload.get("target_count") or config["target_count"]))
            block_ratio = block_index / total_blocks
            row_ratio = row_count / target_count
            qa_ratio = min(1.0, max(row_ratio, block_ratio * 0.65))
            update_job(job_id, progress=min(95, int(55 + qa_ratio * 40)))

            source = Path(payload.get("source", "")).name
            page = payload.get("page") or "unknown"
            planned_items = payload.get("planned_items", 0)
            if event == "qa_chunk_start":
                log_job(job_id, f"QA 生成中：第 {block_index}/{total_blocks} 块，本块计划 {planned_items} 题，已产出 {row_count}/{target_count} 题，来源 {source}，页/位置 {page}")
            elif event == "qa_chunk_done":
                log_job(job_id, f"QA 块完成：第 {block_index}/{total_blocks} 块，本块计划 {planned_items} 题，累计 {row_count}/{target_count} 题，候选 {payload.get('raw_count', 0)} 条，拒绝累计 {payload.get('rejected_count', 0)}")
            else:
                log_job(job_id, f"QA 跳过块：第 {block_index}/{total_blocks} 块，{payload.get('reason', '')}")

        args = SimpleNamespace(
            input=builder_inputs,
            output=str(output_path),
            target_count=config["target_count"],
            items_per_chunk=config["items_per_chunk"],
            max_per_source=config["max_per_source"],
            negative_count=config["negative_count"],
            chunk_chars=1400,
            min_chunk_chars=180,
            chunk_overlap=120,
            min_doc_chars=80,
            backend=config["backend"],
            model=config["model"],
            base_url=config["base_url"],
            timeout=180,
            allow_heuristic=config["allow_heuristic"],
            shuffle_chunks=True,
            seed=42,
            max_zip_depth=3,
            progress_callback=qa_progress,
        )
        check_cancelled(job_id)
        rows, report = build_dataset(args)
        check_cancelled(job_id)
        report["mineru"] = mineru_report
        report["corpus_path"] = str(corpus_path)
        report["builder_inputs"] = builder_inputs
        manifest_path = write_asset_manifest(
            asset_dir,
            dataset_name=config["dataset_name"],
            job_id=job_id,
            original_paths=original_asset_paths,
            builder_inputs=[Path(p) for p in builder_inputs],
            output_path=output_path,
            corpus_path=corpus_path,
            mineru_report=mineru_report,
            config=config,
        )
        report["assets_dir"] = str(asset_dir)
        report["assets_manifest"] = str(manifest_path)
        report["deepseekmine_chunk_sidecars"] = deepseekmine_chunk_sidecars
        write_jsonl(output_path, rows)
        report_path = output_path.with_suffix(output_path.suffix + ".report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        stopped_reason = str(report.get("stopped_reason") or "")
        update_job(
            job_id,
            status="error" if stopped_reason else "done",
            progress=100,
            output_path=str(output_path),
            report_path=str(report_path),
            output_url=f"/download?path={quote(str(output_path))}",
            report_url=f"/download?path={quote(str(report_path))}",
            corpus_path=str(corpus_path),
            corpus_url=f"/download?path={quote(str(corpus_path))}",
            assets_dir=str(asset_dir),
            assets_manifest=str(manifest_path),
        )
        if stopped_reason:
            log_job(job_id, f"API 额度不足/请求超限，任务已保护性中止；已保存部分结果 {len(rows)} 条 QA 到 {output_path}")
        else:
            log_job(job_id, f"完成：{len(rows)} 条 QA；资产已保存到 {asset_dir}")
    except JobCancelled:
        update_job(job_id, status="cancelled", progress=100)
        log_job(job_id, "任务已取消")
    except Exception as exc:
        update_job(job_id, status="error", progress=100)
        log_job(job_id, f"错误：{exc}")
        if "无法连接 deepseekmine 服务" not in str(exc):
            log_job(job_id, traceback.format_exc())


def run_eval_job(job_id: str, config: dict) -> None:
    try:
        update_job(job_id, status="running", progress=5)
        check_cancelled(job_id)
        ensure_llm_config()
        log_job(job_id, f"真实测评数据集：{config['dataset_name']}")
        log_job(job_id, f"DeepLocals：{config['api_base']}")
        log_job(job_id, f"知识库标签：{config['knowledge_label']}")

        result_dir = APP_ROOT / "result-zh"
        result_dir.mkdir(exist_ok=True)
        output_path = result_dir / f"deepseekmine_{config['mode']}_{config['dataset_name']}_{int(time.time())}.json"

        progress_count = {"n": 0}

        def progress(message: str) -> None:
            check_cancelled(job_id)
            progress_count["n"] += 1
            log_job(job_id, message)
            current = min(95, 10 + progress_count["n"] * 3)
            update_job(job_id, progress=current)

        report = run_eval(
            EvalOptions(
                dataset_name=config["dataset_name"],
                dataset_path=config["dataset_path"],
                corpus_path=config["corpus_path"],
                api_base=config["api_base"],
                knowledge_label=config["knowledge_label"],
                mode=config["mode"],
                limit=0,
                output=str(output_path),
                include_prompts=True,
                timeout=900,
            ),
            progress=progress,
        )
        check_cancelled(job_id)
        summary = report.get("summary", {})
        stopped_reason = str(report.get("stopped_reason") or "")
        update_job(
            job_id,
            status="error" if stopped_reason else "done",
            progress=100,
            report_path=str(output_path),
            report_url=f"/download?path={quote(str(output_path))}",
        )
        if stopped_reason:
            log_job(job_id, f"API 额度不足/请求超限，真实测评已保护性中止；部分报告已保存到 {output_path}")
        else:
            log_job(job_id, f"真实测评完成：证据命中率 {summary.get('evidence_hit_rate', 0) * 100:.2f}%")
        if config["mode"] == "qa":
            log_job(job_id, f"问答准确率 {summary.get('qa_accuracy', 0) * 100:.2f}% ({summary.get('qa_correct', 0)}/{summary.get('qa_total', 0)})")
    except JobCancelled:
        update_job(job_id, status="cancelled", progress=100)
        log_job(job_id, "任务已取消")
    except Exception as exc:
        update_job(job_id, status="error", progress=100)
        log_job(job_id, f"错误：{exc}")
        if "无法连接 deepseekmine 服务" not in str(exc):
            log_job(job_id, traceback.format_exc())


def normalize_model_id(value: str) -> str:
    text = str(value or "").strip()
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.lower()


def run_compare_eval_job(job_id: str, config: dict) -> None:
    original_load_llm_config = deep_eval.load_llm_config
    try:
        update_job(job_id, status="running", progress=5)
        check_cancelled(job_id)
        saved_config = load_llm_config()
        provider_id = str(config.get("deeplocal_provider") or saved_config.provider_id or "siliconflow").strip()
        answer_config = saved_config if saved_config.provider_id == provider_id else default_provider_config(provider_id)
        if config.get("deeplocal_api_url"):
            answer_config.api_url = str(config["deeplocal_api_url"]).strip()
        if config.get("deeplocal_model"):
            answer_config.model = str(config["deeplocal_model"]).strip()
            if answer_config.model and answer_config.model not in answer_config.models:
                answer_config.models.insert(0, answer_config.model)
        if config.get("deeplocal_api_key"):
            answer_config.api_key = str(config["deeplocal_api_key"]).strip()
        elif answer_config.provider_id == "siliconflow" and os.environ.get("SILICONFLOW_API_KEY"):
            answer_config.api_key = os.environ["SILICONFLOW_API_KEY"].strip()
        if answer_config.provider_id != "ollama" and not answer_config.api_key:
            raise RuntimeError("DeepLocals 回答模型缺少 API key，请在平台对比表单填写，或先在 API 设置里保存当前模型配置。")
        cherry_api_key = str(config.get("cherry_api_key") or "").strip() or os.environ.get("CHERRY_API_KEY", "").strip()
        if not cherry_api_key:
            raise RuntimeError("Cherry API key 缺失，请在平台对比表单填写，或设置服务进程环境变量 CHERRY_API_KEY。")
        if not str(config.get("cherry_knowledge_base_id") or "").strip():
            raise RuntimeError("Cherry Knowledge Base ID 不能为空。")

        deep_eval.load_llm_config = lambda: answer_config
        if normalize_model_id(config["cherry_model"]) != normalize_model_id(answer_config.model):
            log_job(job_id, f"提醒：DeepLocals 当前模型是 {answer_config.model}，Cherry 模型是 {config['cherry_model']}，请确认它们对应同一个底层模型。")
        log_job(job_id, f"平台对比数据集：{config['dataset_name']}")
        log_job(job_id, f"DeepLocals：{config['deepseekmine_api_base']} / {answer_config.provider_name} / {answer_config.model}")
        log_job(job_id, f"Cherry Studio：{config['cherry_api_base']} / {config['cherry_model']} / 检索 {config['document_count']} 块")
        log_job(job_id, f"题目数量：{'全量' if not config['limit'] else config['limit']}")

        result_dir = APP_ROOT / "result-zh"
        result_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"platform_compare_{config['dataset_name']}_{stamp}"
        deep_report_path = result_dir / f"{prefix}_deeplocal.json"
        cherry_report_path = result_dir / f"{prefix}_cherry.json"
        summary_path = result_dir / f"{prefix}_summary.json"
        csv_path = result_dir / f"{prefix}_summary.csv"
        progress_count = {"n": 0}
        progress_lock = threading.Lock()

        def progress(platform: str, message: str) -> None:
            check_cancelled(job_id)
            with progress_lock:
                progress_count["n"] += 1
                current = min(95, 8 + progress_count["n"] * 2)
            log_job(job_id, f"[{platform}] {message}")
            update_job(job_id, progress=current)

        tasks = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            tasks[
                executor.submit(
                    run_eval,
                    EvalOptions(
                        dataset_name=config["dataset_name"],
                        dataset_path=config["dataset_path"],
                        corpus_path=config["corpus_path"],
                        api_base=config["deepseekmine_api_base"],
                        knowledge_label=f"rag_eval_compare_{config['dataset_name']}_{job_id}_{int(time.time())}",
                        mode="qa",
                        limit=config["limit"],
                        output=str(deep_report_path),
                        include_prompts=False,
                        timeout=1200,
                    ),
                    progress=lambda msg: progress("DeepLocals", msg),
                )
            ] = "DeepLocals"
            tasks[
                executor.submit(
                    run_cherry_eval,
                    CherryEvalOptions(
                        dataset_name=config["dataset_name"],
                        dataset_path=config["dataset_path"],
                        api_base=config["cherry_api_base"],
                        api_key=cherry_api_key,
                        knowledge_base_ids=config["cherry_knowledge_base_id"],
                        model=config["cherry_model"],
                        document_count=config["document_count"],
                        limit=config["limit"],
                        output=str(cherry_report_path),
                        timeout=1200,
                        judge_mode="rule_then_model",
                    ),
                    progress=lambda msg: progress("Cherry", msg),
                )
            ] = "Cherry Studio"

            reports: dict[str, dict] = {}
            for future in as_completed(tasks):
                check_cancelled(job_id)
                platform = tasks[future]
                reports[platform] = future.result()
                log_job(job_id, f"{platform} 完成")

        rows = [
            comparison_row("DeepLocals", reports["DeepLocals"]),
            comparison_row("Cherry Studio", reports["Cherry Studio"]),
        ]
        winner = max(rows, key=lambda item: item["qa_accuracy_percent"])
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset": config["dataset_name"],
            "limit": config["limit"],
            "winner_by_qa_accuracy": winner["platform"],
            "rows": rows,
        }
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        write_comparison_csv(csv_path, rows)

        update_job(
            job_id,
            status="done",
            progress=100,
            report_path=str(summary_path),
            report_url=f"/download?path={quote(str(summary_path))}",
            summary_path=str(summary_path),
            summary_url=f"/download?path={quote(str(summary_path))}",
            csv_path=str(csv_path),
            csv_url=f"/download?path={quote(str(csv_path))}",
        )
        for row in rows:
            log_job(
                job_id,
                f"{row['platform']}：QA {row['qa_accuracy_percent']:.2f}% ({row['qa_correct']}/{row['qa_total']})，"
                f"证据命中 {row['evidence_hit_rate_percent']:.2f}%，MRR {row['mrr']:.4f}",
            )
        log_job(job_id, f"胜出平台：{winner['platform']}")
        log_job(job_id, f"对比摘要：{summary_path}")
        log_job(job_id, f"CSV：{csv_path}")
    except JobCancelled:
        update_job(job_id, status="cancelled", progress=100)
        log_job(job_id, "任务已取消")
    except Exception as exc:
        update_job(job_id, status="error", progress=100)
        log_job(job_id, f"错误：{exc}")
        log_job(job_id, traceback.format_exc())
    finally:
        deep_eval.load_llm_config = original_load_llm_config


def write_eval_corpus(corpus_path: Path, input_paths: list[Path]) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    for path in input_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            text = ""
        if not text:
            continue
        parts.append(f"# Source: {path.name}\n\n{text}")
    if not parts:
        raise RuntimeError("无法写出评测 corpus：解析文本为空")
    corpus_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")


def prepare_dataset_asset_dir(dataset_name: str) -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    asset_dir = ASSETS_DIR / sanitize_dataset_name(dataset_name)
    resolved_root = ASSETS_DIR.resolve()
    resolved_asset = asset_dir.resolve()
    if not str(resolved_asset).lower().startswith(str(resolved_root).lower()):
        raise RuntimeError(f"unsafe asset path: {asset_dir}")
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    return asset_dir


def copy_files_to_dir(paths: list[Path], target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        target = unique_path(target_dir / sanitize_filename(path.name))
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def copy_mineru_result_to_assets(result, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    markdown_source = Path(result.markdown_path)
    markdown_target = unique_path(target_dir / sanitize_filename(markdown_source.name))
    shutil.copy2(markdown_source, markdown_target)

    if getattr(result, "metadata_path", None):
        metadata_source = Path(result.metadata_path)
        if metadata_source.exists():
            shutil.copy2(metadata_source, markdown_target.with_suffix(markdown_target.suffix + ".meta.json"))
    if getattr(result, "combined_json_path", None):
        combined_source = Path(result.combined_json_path)
        if combined_source.exists():
            shutil.copy2(combined_source, markdown_target.with_suffix(markdown_target.suffix + ".mineru.json"))

    raw_zip_dir = target_dir / "raw_zips"
    raw_json_dir = target_dir / "raw_json"
    for source in getattr(result, "zip_paths", None) or []:
        path = Path(source)
        if path.exists():
            copy_files_to_dir([path], raw_zip_dir)
    for source in getattr(result, "json_paths", None) or []:
        path = Path(source)
        if path.exists():
            copy_files_to_dir([path], raw_json_dir)
    return markdown_target


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mineru_cache_dir(digest: str) -> Path:
    if not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise ValueError(f"invalid sha256 digest: {digest}")
    return MINERU_CACHE_DIR / digest[:2] / digest


def load_mineru_cache_entry(digest: str) -> dict | None:
    manifest_path = mineru_cache_dir(digest) / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        entry = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    markdown_path = Path(entry.get("markdown_path", ""))
    semantic_json_path = Path(entry.get("semantic_json_path", ""))
    metadata_path = Path(entry.get("metadata_path", ""))
    if markdown_path.exists() and semantic_json_path.exists() and metadata_path.exists():
        return entry
    return None


def store_mineru_result_in_cache(result, digest: str, original_path: Path) -> Path:
    cache_dir = mineru_cache_dir(digest)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    original_copies = copy_files_to_dir([original_path], cache_dir / "original")
    markdown_path = copy_mineru_result_to_assets(result, cache_dir / "mineru")
    metadata_path = markdown_path.with_suffix(markdown_path.suffix + ".meta.json")
    semantic_json_path = markdown_path.with_suffix(markdown_path.suffix + ".mineru.json")
    manifest = {
        "schema": "rageval-mineru-cache-v1",
        "file_sha256": digest,
        "original_name": original_path.name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_files": original_copies,
        "markdown_path": str(markdown_path),
        "metadata_path": str(metadata_path),
        "semantic_json_path": str(semantic_json_path),
        "raw_json_files": [str(path) for path in sorted((cache_dir / "mineru" / "raw_json").glob("*.json"))],
        "raw_zip_files": [str(path) for path in sorted((cache_dir / "mineru" / "raw_zips").glob("*.zip"))],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown_path


def copy_cached_mineru_to_assets(cache_entry: dict, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    markdown_source = Path(cache_entry["markdown_path"])
    markdown_target = unique_path(target_dir / sanitize_filename(markdown_source.name))
    shutil.copy2(markdown_source, markdown_target)

    metadata_source = Path(cache_entry["metadata_path"])
    semantic_source = Path(cache_entry["semantic_json_path"])
    if metadata_source.exists():
        shutil.copy2(metadata_source, markdown_target.with_suffix(markdown_target.suffix + ".meta.json"))
    if semantic_source.exists():
        shutil.copy2(semantic_source, markdown_target.with_suffix(markdown_target.suffix + ".mineru.json"))

    for raw in cache_entry.get("raw_json_files") or []:
        path = Path(raw)
        if path.exists():
            copy_files_to_dir([path], target_dir / "raw_json")
    for raw in cache_entry.get("raw_zip_files") or []:
        path = Path(raw)
        if path.exists():
            copy_files_to_dir([path], target_dir / "raw_zips")
    return markdown_target


def write_deepseekmine_compatible_chunk_sidecar(markdown_path: Path, dataset_name: str) -> Path | None:
    try:
        return write_chunk_sidecar(markdown_path, sanitize_dataset_name(dataset_name))
    except Exception:
        return None


def write_asset_manifest(
    asset_dir: Path,
    dataset_name: str,
    job_id: str,
    original_paths: list[str],
    builder_inputs: list[Path],
    output_path: Path,
    corpus_path: Path,
    mineru_report: list[dict],
    config: dict,
) -> Path:
    safe_config = {
        key: value
        for key, value in config.items()
        if "key" not in key.lower() and "token" not in key.lower()
    }
    manifest = {
        "dataset_name": dataset_name,
        "job_id": job_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_files": original_paths,
        "mineru_markdown_files": [str(path) for path in builder_inputs if path.suffix.lower() in {".md", ".markdown"}],
        "mineru_metadata_files": [
            str(path.with_suffix(path.suffix + ".meta.json"))
            for path in builder_inputs
            if path.with_suffix(path.suffix + ".meta.json").exists()
        ],
        "mineru_semantic_json_files": [
            str(path.with_suffix(path.suffix + ".mineru.json"))
            for path in builder_inputs
            if path.with_suffix(path.suffix + ".mineru.json").exists()
        ],
        "deepseekmine_chunk_files": [
            str(path.with_suffix(path.suffix + ".deepseekmine_chunks.json"))
            for path in builder_inputs
            if path.with_suffix(path.suffix + ".deepseekmine_chunks.json").exists()
        ],
        "raw_mineru_json_files": [str(path) for path in sorted((asset_dir / "mineru" / "raw_json").glob("*.json"))],
        "raw_mineru_zip_files": [str(path) for path in sorted((asset_dir / "mineru" / "raw_zips").glob("*.zip"))],
        "mineru_cache": {
            "root": str(MINERU_CACHE_DIR),
            "file_sha256": [
                item.get("file_sha256")
                for item in mineru_report
                if item.get("file_sha256")
            ],
        },
        "dataset_path": str(output_path),
        "corpus_path": str(corpus_path),
        "mineru_report": mineru_report,
        "config": safe_config,
    }
    manifest_path = asset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def ensure_llm_config() -> None:
    config = load_llm_config()
    if config.provider_id == "ollama":
        return
    if config.api_key:
        return
    fallback = default_provider_config(config.provider_id)
    if fallback.api_key:
        return
    raise RuntimeError(f"{config.provider_name} API key is missing. Please configure it in API Settings or set the provider environment variable.")


def ensure_mineru_key() -> str:
    env_key = os.environ.get("MINERU_API_TOKEN", "").strip()
    if env_key:
        return env_key
    if MINERU_KEY_FILE.exists():
        key = MINERU_KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["MINERU_API_TOKEN"] = key
            return key
    raise RuntimeError("MinerU API token is missing. Set MINERU_API_TOKEN or create E:\\RAG\\.mineru_api_key")


def sanitize_dataset_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80].strip("._-")
    return value or "custom_rag_eval"


def sanitize_filename(value: str) -> str:
    value = value.replace("\\", "_").replace("/", "_").strip()
    value = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", value)
    return value or f"upload_{uuid.uuid4().hex[:8]}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"cannot allocate unique path for {path}")


def is_download_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
        allowed_roots = [DATA_DIR.resolve(), RUNS_DIR.resolve(), RESULT_DIR.resolve()]
        return any(str(resolved).lower().startswith(str(root).lower()) for root in allowed_roots)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MINERU_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RAGDatasetUIHandler)
    print(f"[ui] RAG dataset UI: http://{args.host}:{args.port}")
    print("[ui] Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
