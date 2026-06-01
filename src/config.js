import path from "node:path";
import { fileURLToPath } from "node:url";

export const APP_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const DATA_DIR = path.join(APP_ROOT, "data");
export const RESULT_ZH_DIR = path.join(APP_ROOT, "result-zh");
export const TEMP_DOCS_DIR = path.join(APP_ROOT, "temp_docs");
export const MIMO_KEY_FILE = path.join(APP_ROOT, ".mimo_api_key");
export const MINERU_KEY_FILE = path.join(APP_ROOT, ".mineru_api_key");
export const LLM_CONFIG_FILE = path.join(APP_ROOT, ".rageval_api_config.json");
export const OLLAMA_CONTEXT_CONFIG_FILE = path.join(APP_ROOT, ".rageval_ollama_context.json");
export const MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1";
export const MIMO_MODEL = "mimo-v2.5-pro";
