from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from rageval.config import LLM_CONFIG_FILE, MIMO_BASE_URL, MIMO_KEY_FILE, MIMO_MODEL, OLLAMA_CONTEXT_CONFIG_FILE


QUOTA_ERROR_PATTERNS = [
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
    "欠费",
]

OLLAMA_CONTEXT_LENGTH_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
DEFAULT_OLLAMA_CONTEXT_LENGTH = 8192


@dataclass
class ProviderPreset:
    id: str
    name: str
    api_url: str
    default_model: str
    models: list[str]
    auth_type: str = "bearer"
    env_key: str = ""
    key_file: str = ""
    description: str = ""
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.95
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMProviderConfig:
    provider_id: str = "mimo"
    provider_name: str = "MiMo"
    api_key: str = ""
    api_url: str = MIMO_BASE_URL
    model: str = MIMO_MODEL
    models: list[str] = field(default_factory=lambda: [MIMO_MODEL])
    auth_type: str = "api-key"
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.95
    enabled: bool = True
    description: str = ""
    extra_body: dict[str, Any] = field(default_factory=lambda: {"thinking": {"type": "disabled"}})


class LLMQuotaExceededError(RuntimeError):
    pass


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "mimo": ProviderPreset(
        id="mimo",
        name="MiMo",
        api_url=MIMO_BASE_URL,
        default_model=MIMO_MODEL,
        models=[MIMO_MODEL],
        auth_type="api-key",
        env_key="MIMO_API_KEY",
        key_file=str(MIMO_KEY_FILE),
        description="小米 MiMo OpenAI-compatible 接口",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.95,
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "siliconflow": ProviderPreset(
        id="siliconflow",
        name="硅基流动",
        api_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen3.5-4B",
        models=[
            "Qwen/Qwen3.5-4B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.5-27B",
            "Qwen/Qwen3-8B",
        ],
        auth_type="bearer",
        env_key="SILICONFLOW_API_KEY",
        description="SiliconFlow OpenAI-compatible Chat Completions",
        max_tokens=4096,
        temperature=0.0,
        top_p=0.95,
        extra_body={"enable_thinking": False},
    ),
    "deepseek": ProviderPreset(
        id="deepseek",
        name="DeepSeek",
        api_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=["deepseek-chat", "deepseek-reasoner"],
        auth_type="bearer",
        env_key="DEEPSEEK_API_KEY",
        description="DeepSeek OpenAI-compatible API",
    ),
    "qwen": ProviderPreset(
        id="qwen",
        name="阿里云百炼",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        models=["qwen-plus", "qwen-turbo", "qwen-max"],
        auth_type="bearer",
        env_key="DASHSCOPE_API_KEY",
        description="DashScope 百炼 OpenAI-compatible mode",
    ),
    "ollama": ProviderPreset(
        id="ollama",
        name="Ollama 本地",
        api_url="http://localhost:11434/v1",
        default_model="qwen3:8b",
        models=["qwen3:8b", "llama3.2", "gpt-oss:20b"],
        auth_type="bearer",
        env_key="OLLAMA_API_KEY",
        description="Ollama local OpenAI-compatible API",
    ),
    "custom": ProviderPreset(
        id="custom",
        name="自定义 API",
        api_url="https://api.example.com/v1",
        default_model="custom-model",
        models=["custom-model"],
        auth_type="bearer",
        env_key="RAGEVAL_CUSTOM_API_KEY",
        description="OpenAI-compatible 自定义接口",
    ),
}


def provider_presets_payload() -> list[dict[str, Any]]:
    return [asdict(preset) for preset in PROVIDER_PRESETS.values()]


def is_quota_error(status_code: int, body: str) -> bool:
    text = str(body or "").lower()
    if status_code == 402:
        return True
    return any(pattern in text for pattern in QUOTA_ERROR_PATTERNS)


def ensure_chat_completions_url(api_url: str) -> str:
    value = str(api_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        return value
    if "open.bigmodel.cn" in value and "/api/paas/v4" not in value:
        return value + "/api/paas/v4/chat/completions"
    if value.endswith("/v1") or value.endswith("/v2") or value.endswith("/api/paas/v4"):
        return value + "/chat/completions"
    if "/v1/" in value and "/chat/completions" not in value:
        return value.replace("/v1/", "/v1/chat/completions/")
    if "/v2/" in value and "/chat/completions" not in value:
        return value.replace("/v2/", "/v2/chat/completions/")
    if "/chat/completions" not in value and "/v1" not in value and "/v2" not in value:
        return value + "/v1/chat/completions"
    return value


def ensure_models_url(api_url: str) -> str:
    value = str(api_url or "").strip().rstrip("/")
    if not value:
        return ""
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    if value.endswith("/models"):
        return value
    if value.endswith("/v1") or value.endswith("/v2") or value.endswith("/api/paas/v4"):
        return value + "/models"
    if "/v1/" in value:
        return value.split("/v1/", 1)[0] + "/v1/models"
    if "/v2/" in value:
        return value.split("/v2/", 1)[0] + "/v2/models"
    if "/v1" not in value and "/v2" not in value:
        return value + "/v1/models"
    return value + "/models"


def ensure_ollama_tags_url(api_url: str) -> str:
    value = str(api_url or "http://localhost:11434/v1").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/models", "/v1", "/api/chat", "/api/tags"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/") + "/api/tags"


def ensure_ollama_chat_url(api_url: str) -> str:
    value = str(api_url or "http://localhost:11434/v1").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/models", "/v1", "/api/chat", "/api/tags"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/") + "/api/chat"


def _read_key_from_file(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def default_provider_config(provider_id: str = "mimo") -> LLMProviderConfig:
    preset = PROVIDER_PRESETS.get(provider_id) or PROVIDER_PRESETS["mimo"]
    key = os.environ.get(preset.env_key, "").strip() or _read_key_from_file(preset.key_file)
    if preset.id == "ollama":
        key = ""
    return LLMProviderConfig(
        provider_id=preset.id,
        provider_name=preset.name,
        api_key=key,
        api_url=preset.api_url,
        model=preset.default_model,
        models=list(preset.models),
        auth_type=preset.auth_type,
        max_tokens=preset.max_tokens,
        temperature=preset.temperature,
        top_p=preset.top_p,
        enabled=True,
        description=preset.description,
        extra_body=dict(preset.extra_body),
    )


def load_llm_config() -> LLMProviderConfig:
    if not LLM_CONFIG_FILE.exists():
        return default_provider_config("mimo")
    try:
        data = json.loads(LLM_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return default_provider_config("mimo")
    if not isinstance(data, dict):
        return default_provider_config("mimo")
    provider_id = str(data.get("provider_id") or data.get("providerId") or "mimo").strip() or "mimo"
    base = default_provider_config(provider_id)
    merged = asdict(base)
    key_map = {
        "providerId": "provider_id",
        "providerName": "provider_name",
        "apiKey": "api_key",
        "apiUrl": "api_url",
        "defaultModel": "model",
        "maxTokens": "max_tokens",
        "topP": "top_p",
        "isEnabled": "enabled",
        "extraBody": "extra_body",
    }
    for key, value in data.items():
        merged[key_map.get(key, key)] = value
    if provider_id == "ollama":
        merged["api_key"] = ""
    if not str(merged.get("api_key") or "").strip():
        preset = PROVIDER_PRESETS.get(provider_id)
        if preset:
            merged["api_key"] = os.environ.get(preset.env_key, "").strip() or _read_key_from_file(preset.key_file)
            if preset.id == "ollama":
                merged["api_key"] = ""
    if not merged.get("models"):
        merged["models"] = [merged.get("model") or base.model]
    return LLMProviderConfig(**{k: v for k, v in merged.items() if k in LLMProviderConfig.__dataclass_fields__})


def save_llm_config(config: LLMProviderConfig) -> None:
    LLM_CONFIG_FILE.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_ollama_context_length(value: Any) -> int:
    try:
        context_length = int(value)
    except (TypeError, ValueError):
        return DEFAULT_OLLAMA_CONTEXT_LENGTH
    if context_length < OLLAMA_CONTEXT_LENGTH_OPTIONS[0] or context_length > OLLAMA_CONTEXT_LENGTH_OPTIONS[-1]:
        return DEFAULT_OLLAMA_CONTEXT_LENGTH
    return context_length


def load_ollama_context_config() -> dict[str, int]:
    if not OLLAMA_CONTEXT_CONFIG_FILE.exists():
        return {"context_length": DEFAULT_OLLAMA_CONTEXT_LENGTH}
    try:
        data = json.loads(OLLAMA_CONTEXT_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"context_length": DEFAULT_OLLAMA_CONTEXT_LENGTH}
    if not isinstance(data, dict):
        return {"context_length": DEFAULT_OLLAMA_CONTEXT_LENGTH}
    return {"context_length": normalize_ollama_context_length(data.get("context_length"))}


def save_ollama_context_config(context_length: Any) -> dict[str, int]:
    config = {"context_length": normalize_ollama_context_length(context_length)}
    OLLAMA_CONTEXT_CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def config_from_form(data: dict[str, Any]) -> LLMProviderConfig:
    provider_id = str(data.get("provider_id") or data.get("providerId") or "mimo").strip() or "mimo"
    base = default_provider_config(provider_id)
    models_text = str(data.get("models") or "").replace("\r", "\n")
    models = [item.strip() for item in re.split(r"[\n,]+", models_text) if item.strip()]
    model = str(data.get("model") or data.get("defaultModel") or base.model).strip() or base.model
    if model and model not in models:
        models.insert(0, model)
    def as_float(name: str, default: float) -> float:
        try:
            return float(data.get(name, default))
        except (TypeError, ValueError):
            return default
    def as_int(name: str, default: int) -> int:
        try:
            return int(data.get(name, default))
        except (TypeError, ValueError):
            return default
    extra_body = dict(base.extra_body)
    if provider_id == "siliconflow":
        raw = str(data.get("enable_thinking") or "").lower()
        extra_body["enable_thinking"] = raw in {"1", "true", "yes", "on"}
    api_key = str(data.get("api_key") or "").strip()
    if provider_id == "ollama":
        api_key = ""
    return LLMProviderConfig(
        provider_id=provider_id,
        provider_name=str(data.get("provider_name") or base.provider_name).strip() or base.provider_name,
        api_key=api_key,
        api_url=str(data.get("api_url") or base.api_url).strip() or base.api_url,
        model=model,
        models=models or list(base.models),
        auth_type=str(data.get("auth_type") or base.auth_type).strip() or base.auth_type,
        max_tokens=as_int("max_tokens", base.max_tokens),
        temperature=as_float("temperature", base.temperature),
        top_p=as_float("top_p", base.top_p),
        enabled=bool(data.get("enabled", True)),
        description=str(data.get("description") or base.description),
        extra_body=extra_body,
    )


def masked_config(config: LLMProviderConfig | None = None) -> dict[str, Any]:
    config = config or load_llm_config()
    payload = asdict(config)
    key = str(payload.get("api_key") or "")
    if key:
        payload["api_key_masked"] = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
        payload["api_key"] = ""
    return payload


def _headers(config: LLMProviderConfig) -> dict[str, str]:
    if config.provider_id == "ollama":
        return {"Content-Type": "application/json"}
    if not config.api_key:
        raise RuntimeError(f"{config.provider_name} API key is missing")
    headers = {"Content-Type": "application/json"}
    if config.auth_type == "api-key":
        headers["api-key"] = config.api_key
    else:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def list_provider_models(config: LLMProviderConfig, timeout: int = 20) -> list[str]:
    models_url = ensure_ollama_tags_url(config.api_url) if config.provider_id == "ollama" else ensure_models_url(config.api_url)
    if not models_url:
        raise RuntimeError("API 地址不能为空")
    resp = requests.get(models_url, headers=_headers(config), timeout=timeout)
    body = resp.text[:800].replace("\n", " ")
    if resp.status_code >= 400:
        raise RuntimeError(f"{config.provider_name} 模型列表获取失败: HTTP {resp.status_code} {body}")
    payload = resp.json()
    data = payload.get("models", []) if config.provider_id == "ollama" else payload.get("data", payload if isinstance(payload, list) else [])
    models: list[str] = []
    if isinstance(data, list):
        for item in data:
            model_id = (item.get("name") or item.get("model") or item.get("id")) if isinstance(item, dict) else item
            model_id = str(model_id or "").strip()
            if model_id and model_id not in models:
                models.append(model_id)
    if not models:
        if config.provider_id == "ollama":
            raise RuntimeError("未检测到本地 Ollama 模型，请先启动 Ollama 并 pull 模型")
        raise RuntimeError(f"{config.provider_name} 未返回可用模型列表")
    return models


def chat_completion(
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    purpose: str,
    config: LLMProviderConfig | None = None,
) -> dict[str, Any]:
    config = config or load_llm_config()
    is_ollama = config.provider_id == "ollama"
    completion_url = ensure_ollama_chat_url(config.api_url) if is_ollama else ensure_chat_completions_url(config.api_url)
    payload: dict[str, Any] = {"model": config.model, "messages": messages, "stream": False}
    if is_ollama:
        payload["think"] = False
        payload["options"] = {"num_ctx": load_ollama_context_config()["context_length"]}
        if purpose.startswith("judge") or purpose == "dataset generation":
            payload["format"] = "json"
    else:
        payload["temperature"] = config.temperature
        payload["top_p"] = config.top_p
    token_budget = max_tokens or config.max_tokens
    if is_ollama:
        payload["options"]["num_predict"] = token_budget
    elif config.provider_id == "mimo":
        payload["max_completion_tokens"] = token_budget
    else:
        payload["max_tokens"] = token_budget
    if not is_ollama:
        payload.update(config.extra_body or {})

    retry_delays = [2, 6, 12]
    last_error = ""
    for attempt in range(len(retry_delays) + 1):
        try:
            resp = requests.post(completion_url, headers=_headers(config), json=payload, timeout=timeout)
            body = resp.text[:800].replace("\n", " ")
            if resp.status_code >= 400 and is_quota_error(resp.status_code, body):
                raise LLMQuotaExceededError(f"{config.provider_name} {purpose} quota exhausted: HTTP {resp.status_code} {body}")
            if resp.status_code == 429:
                last_error = f"{config.provider_name} {purpose} request limit: HTTP 429 {body[:300]}"
                if attempt < len(retry_delays):
                    time.sleep(retry_delays[attempt])
                    continue
                raise LLMQuotaExceededError(last_error)
            if resp.status_code in {500, 502, 503, 504} and attempt < len(retry_delays):
                last_error = f"{config.provider_name} {purpose} HTTP {resp.status_code}: {body[:300]}"
                time.sleep(retry_delays[attempt])
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"{config.provider_name} {purpose} failed: HTTP {resp.status_code} {body}")
            data = resp.json()
            if is_ollama:
                content = str(((data.get("message") or {}).get("content") or "")).strip()
                return {
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": int(data.get("prompt_eval_count", 0) or 0) + int(data.get("eval_count", 0) or 0),
                    },
                    "ollama": data,
                }
            return data
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{config.provider_name} {purpose} request error: {exc}"
            if attempt < len(retry_delays):
                time.sleep(retry_delays[attempt])
                continue
            raise RuntimeError(last_error) from exc
    raise RuntimeError(last_error or f"{config.provider_name} {purpose} failed")


def message_text(data: dict[str, Any]) -> str:
    message = data.get("choices", [{}])[0].get("message", {}) or {}
    return str(message.get("content") or message.get("reasoning_content") or "").strip()
