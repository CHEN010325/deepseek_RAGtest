import fs from "node:fs/promises";
import { LLM_CONFIG_FILE, MIMO_BASE_URL, MIMO_KEY_FILE, MIMO_MODEL, OLLAMA_CONTEXT_CONFIG_FILE } from "./config.js";
import { pathExists, readJson } from "./utils/files.js";

export class LLMQuotaExceededError extends Error {
  constructor(message) {
    super(message);
    this.name = "LLMQuotaExceededError";
  }
}

const QUOTA_ERROR_PATTERNS = [
  "insufficient_quota",
  "quota exceeded",
  "quota_exceeded",
  "quota exhausted",
  "insufficient balance",
  "out of credits",
  "billing",
  "payment required",
  "account balance",
  "resource exhausted",
  "request limit",
  "余额不足",
  "额度不足",
  "配额不足",
  "账户余额",
  "资源耗尽",
  "欠费"
];

export const PROVIDER_PRESETS = {
  mimo: {
    id: "mimo",
    name: "MiMo",
    api_url: MIMO_BASE_URL,
    default_model: MIMO_MODEL,
    models: [MIMO_MODEL],
    auth_type: "api-key",
    env_key: "MIMO_API_KEY",
    key_file: MIMO_KEY_FILE,
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: { thinking: { type: "disabled" } }
  },
  siliconflow: {
    id: "siliconflow",
    name: "SiliconFlow",
    api_url: "https://api.siliconflow.cn/v1",
    default_model: "Qwen/Qwen3.5-4B",
    models: ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3-8B"],
    auth_type: "bearer",
    env_key: "SILICONFLOW_API_KEY",
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: { enable_thinking: false }
  },
  deepseek: {
    id: "deepseek",
    name: "DeepSeek",
    api_url: "https://api.deepseek.com/v1",
    default_model: "deepseek-chat",
    models: ["deepseek-chat", "deepseek-reasoner"],
    auth_type: "bearer",
    env_key: "DEEPSEEK_API_KEY",
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: {}
  },
  qwen: {
    id: "qwen",
    name: "DashScope",
    api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_model: "qwen-plus",
    models: ["qwen-plus", "qwen-turbo", "qwen-max"],
    auth_type: "bearer",
    env_key: "DASHSCOPE_API_KEY",
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: {}
  },
  ollama: {
    id: "ollama",
    name: "Ollama",
    api_url: "http://localhost:11434/v1",
    default_model: "qwen3:8b",
    models: ["qwen3:8b", "llama3.2", "gpt-oss:20b"],
    auth_type: "bearer",
    env_key: "OLLAMA_API_KEY",
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: {}
  },
  custom: {
    id: "custom",
    name: "Custom API",
    api_url: "https://api.example.com/v1",
    default_model: "custom-model",
    models: ["custom-model"],
    auth_type: "bearer",
    env_key: "RAGEVAL_CUSTOM_API_KEY",
    max_tokens: 4096,
    temperature: 0.0,
    top_p: 0.95,
    extra_body: {}
  }
};

async function readKeyFromFile(filePath) {
  if (!filePath || !(await pathExists(filePath))) return "";
  return (await fs.readFile(filePath, "utf8")).trim();
}

export async function defaultProviderConfig(providerId = "mimo") {
  const preset = PROVIDER_PRESETS[providerId] ?? PROVIDER_PRESETS.mimo;
  let apiKey = (process.env[preset.env_key] ?? "").trim() || (await readKeyFromFile(preset.key_file));
  if (preset.id === "ollama") apiKey = "";
  return {
    provider_id: preset.id,
    provider_name: preset.name,
    api_key: apiKey,
    api_url: preset.api_url,
    model: preset.default_model,
    models: [...preset.models],
    auth_type: preset.auth_type,
    max_tokens: preset.max_tokens,
    temperature: preset.temperature,
    top_p: preset.top_p,
    enabled: true,
    description: preset.description ?? "",
    extra_body: { ...(preset.extra_body ?? {}) }
  };
}

export async function loadLlmConfig() {
  const data = await readJson(LLM_CONFIG_FILE, null);
  if (!data || typeof data !== "object") return defaultProviderConfig("mimo");
  const providerId = String(data.provider_id ?? data.providerId ?? "mimo").trim() || "mimo";
  const base = await defaultProviderConfig(providerId);
  const keyMap = {
    providerId: "provider_id",
    providerName: "provider_name",
    apiKey: "api_key",
    apiUrl: "api_url",
    defaultModel: "model",
    maxTokens: "max_tokens",
    topP: "top_p",
    isEnabled: "enabled",
    extraBody: "extra_body"
  };
  const merged = { ...base };
  for (const [key, value] of Object.entries(data)) {
    merged[keyMap[key] ?? key] = value;
  }
  if (providerId === "ollama") merged.api_key = "";
  if (!String(merged.api_key ?? "").trim()) {
    const preset = PROVIDER_PRESETS[providerId];
    if (preset) {
      merged.api_key = (process.env[preset.env_key] ?? "").trim() || (await readKeyFromFile(preset.key_file));
      if (preset.id === "ollama") merged.api_key = "";
    }
  }
  if (!Array.isArray(merged.models) || !merged.models.length) merged.models = [merged.model || base.model];
  return merged;
}

export function ensureChatCompletionsUrl(apiUrl) {
  let value = String(apiUrl ?? "").trim().replace(/\/+$/u, "");
  if (!value) return "";
  if (value.endsWith("/chat/completions")) return value;
  if (value.includes("open.bigmodel.cn") && !value.includes("/api/paas/v4")) {
    return `${value}/api/paas/v4/chat/completions`;
  }
  if (/(\/v1|\/v2|\/api\/paas\/v4)$/u.test(value)) return `${value}/chat/completions`;
  if (value.includes("/v1/") && !value.includes("/chat/completions")) return value.replace("/v1/", "/v1/chat/completions/");
  if (value.includes("/v2/") && !value.includes("/chat/completions")) return value.replace("/v2/", "/v2/chat/completions/");
  if (!value.includes("/chat/completions") && !value.includes("/v1") && !value.includes("/v2")) {
    return `${value}/v1/chat/completions`;
  }
  return value;
}

function ensureOllamaChatUrl(apiUrl) {
  let value = String(apiUrl || "http://localhost:11434/v1").trim().replace(/\/+$/u, "");
  for (const suffix of ["/v1/chat/completions", "/v1/models", "/v1", "/api/chat", "/api/tags"]) {
    if (value.endsWith(suffix)) {
      value = value.slice(0, -suffix.length);
      break;
    }
  }
  return `${value.replace(/\/+$/u, "")}/api/chat`;
}

async function loadOllamaContextLength() {
  const data = await readJson(OLLAMA_CONTEXT_CONFIG_FILE, null);
  const raw = Number.parseInt(String(data?.context_length ?? 8192), 10);
  return Number.isInteger(raw) ? raw : 8192;
}

function isQuotaError(statusCode, body) {
  const text = String(body ?? "").toLowerCase();
  if (statusCode === 402) return true;
  return QUOTA_ERROR_PATTERNS.some((pattern) => text.includes(pattern));
}

function headersFor(config) {
  if (config.provider_id === "ollama") return { "Content-Type": "application/json" };
  if (!config.api_key) throw new Error(`${config.provider_name} API key is missing`);
  const headers = { "Content-Type": "application/json" };
  if (config.auth_type === "api-key") headers["api-key"] = config.api_key;
  else headers.Authorization = `Bearer ${config.api_key}`;
  return headers;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function chatCompletion(messages, maxTokens, timeout, purpose, config = null) {
  const cfg = config ?? (await loadLlmConfig());
  const isOllama = cfg.provider_id === "ollama";
  const completionUrl = isOllama ? ensureOllamaChatUrl(cfg.api_url) : ensureChatCompletionsUrl(cfg.api_url);
  const payload = { model: cfg.model, messages, stream: false };
  const tokenBudget = maxTokens || cfg.max_tokens || 4096;
  if (isOllama) {
    payload.think = false;
    payload.options = { num_ctx: await loadOllamaContextLength(), num_predict: tokenBudget };
    if (String(purpose).startsWith("judge") || purpose === "dataset generation") payload.format = "json";
  } else {
    payload.temperature = Number(cfg.temperature ?? 0);
    payload.top_p = Number(cfg.top_p ?? 0.95);
    if (cfg.provider_id === "mimo") payload.max_completion_tokens = tokenBudget;
    else payload.max_tokens = tokenBudget;
    Object.assign(payload, cfg.extra_body ?? {});
  }

  const retryDelays = [2000, 6000, 12000];
  let lastError = "";
  for (let attempt = 0; attempt <= retryDelays.length; attempt += 1) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout * 1000);
      const response = await fetch(completionUrl, {
        method: "POST",
        headers: headersFor(cfg),
        body: JSON.stringify(payload),
        signal: controller.signal
      }).finally(() => clearTimeout(timer));
      const body = await response.text();
      const bodyPreview = body.slice(0, 800).replace(/\n/gu, " ");
      if (response.status >= 400 && isQuotaError(response.status, bodyPreview)) {
        throw new LLMQuotaExceededError(`${cfg.provider_name} ${purpose} quota exhausted: HTTP ${response.status} ${bodyPreview}`);
      }
      if (response.status === 429) {
        lastError = `${cfg.provider_name} ${purpose} request limit: HTTP 429 ${bodyPreview.slice(0, 300)}`;
        if (attempt < retryDelays.length) {
          await sleep(retryDelays[attempt]);
          continue;
        }
        throw new LLMQuotaExceededError(lastError);
      }
      if ([500, 502, 503, 504].includes(response.status) && attempt < retryDelays.length) {
        lastError = `${cfg.provider_name} ${purpose} HTTP ${response.status}: ${bodyPreview.slice(0, 300)}`;
        await sleep(retryDelays[attempt]);
        continue;
      }
      if (response.status >= 400) {
        throw new Error(`${cfg.provider_name} ${purpose} failed: HTTP ${response.status} ${bodyPreview}`);
      }
      const data = JSON.parse(body || "{}");
      if (isOllama) {
        const content = String(data.message?.content ?? "").trim();
        return {
          choices: [{ message: { role: "assistant", content } }],
          usage: {
            prompt_tokens: data.prompt_eval_count ?? 0,
            completion_tokens: data.eval_count ?? 0,
            total_tokens: Number(data.prompt_eval_count ?? 0) + Number(data.eval_count ?? 0)
          },
          ollama: data
        };
      }
      return data;
    } catch (error) {
      if (error instanceof LLMQuotaExceededError) throw error;
      lastError = `${cfg.provider_name} ${purpose} request error: ${error.message}`;
      if (attempt < retryDelays.length) {
        await sleep(retryDelays[attempt]);
        continue;
      }
      throw new Error(lastError);
    }
  }
  throw new Error(lastError || `${cfg.provider_name} ${purpose} failed`);
}

export function messageText(data) {
  const message = data?.choices?.[0]?.message ?? {};
  return String(message.content ?? message.reasoning_content ?? "").trim();
}
