"""Minimal DeepSeek OpenAI-compatible client for no-leakage Agent reports."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_ENV = Path("/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/appkey.env")
DEFAULT_BASE_URL = "https://api.deepseek.com"


def load_env_file(path: str | Path = DEFAULT_ENV) -> dict[str, str]:
    """Load KEY=VALUE pairs without printing or persisting secrets."""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value:
            loaded[key] = value
            os.environ.setdefault(key, value)
    return loaded


class DeepSeekClient:
    """DeepSeek chat client using the OpenAI-compatible chat/completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        env_file: str | Path = DEFAULT_ENV,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
    ) -> None:
        load_env_file(env_file)
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK")
        if not self.api_key:
            raise ValueError(f"DeepSeek API key missing. Set DEEPSEEK or DEEPSEEK_API_KEY in {env_file}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_output: bool = False,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.error("DeepSeek request failed: status=%s body=%s", resp.status_code, resp.text[:500])
            raise
        if "error" in data:
            raise RuntimeError(f"DeepSeek API error: {data['error']}")
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content.strip():
            raise ValueError("DeepSeek response did not include final message content; reasoning_content is not used as Agent output.")
        return content.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for i, char in enumerate(cleaned[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start : i + 1])
    raise ValueError(f"No JSON object found in model output: {text[:200]}")
