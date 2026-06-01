from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
ASSETS_DIR = DATA_DIR / "assets"
MINERU_CACHE_DIR = DATA_DIR / "mineru_cache"
RUNS_DIR = APP_ROOT / "temp_docs" / "ui_runs"
RESULT_ZH_DIR = APP_ROOT / "result-zh"
TEMP_DOCS_DIR = APP_ROOT / "temp_docs"
MIMO_KEY_FILE = APP_ROOT / ".mimo_api_key"
MINERU_KEY_FILE = APP_ROOT / ".mineru_api_key"
LLM_CONFIG_FILE = APP_ROOT / ".rageval_api_config.json"
OLLAMA_CONTEXT_CONFIG_FILE = APP_ROOT / ".rageval_ollama_context.json"
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"
