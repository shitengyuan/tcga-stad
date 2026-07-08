"""
friday_client.py
════════════════
美团 Friday 大模型平台 OpenAI 兼容客户端。

接口: https://aigc.sankuai.com/v1/openai/native/chat/completions
鉴权: Header Authorization: <AppID>

支持模型:
  - glm-5.2          文本推理 (带 reasoning, 用于 agent 决策)
  - glm-4.6          文本 (备选)
  - glm-4v-plus      视觉理解 (图像 -> 文本)
  - qwen-vl-max-latest 视觉 (备选, 需权限)

用法:
  from src.friday_client import FridayClient
  fc = FridayClient(app_id="...")
  text = fc.chat("用一句话描述MSI-H胃癌的形态特征", model="glm-5.2")
  desc = fc.vision(image_url_or_path, "这张病理图有什么特征?", model="glm-4v-plus")
"""
from __future__ import annotations
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://aigc.sankuai.com/v1/openai/native/chat/completions"


class FridayClient:
    """Friday API 客户端。

    Parameters
    ----------
    app_id : str
        Friday 平台申请的 AppID (鉴权用)
    timeout : int
        请求超时秒数
    """

    def __init__(self, app_id: Optional[str] = None, timeout: int = 120):
        self.app_id = app_id or os.environ.get("FRIDAY_APP_ID")
        if not self.app_id:
            raise ValueError("需提供 app_id 或设环境变量 FRIDAY_APP_ID")
        self.timeout = timeout

    # ── 文本对话 ─────────────────────────────────────────────

    def chat(self, prompt: str, model: str = "glm-5.2",
             system: Optional[str] = None, max_tokens: int = 2048,
             temperature: float = 0.3) -> str:
        """文本对话, 返回 assistant 回复内容。"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._call(messages, model, max_tokens, temperature)

    # ── 视觉理解 ─────────────────────────────────────────────

    def vision(self, image: str, prompt: str, model: str = "glm-4v-plus",
               max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """图像理解。

        Parameters
        ----------
        image : str
            图像路径(本地) 或 URL(http/https)。本地路径会被 base64 编码。
        prompt : str
            对图像的提问。
        model : str
            视觉模型, 默认 glm-4v-plus。
        """
        image_url = self._resolve_image(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        return self._call(messages, model, max_tokens, temperature)

    # ── 内部 ─────────────────────────────────────────────────

    def _call(self, messages, model, max_tokens, temperature) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": self.app_id,
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(BASE_URL, headers=headers, json=payload,
                                 timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Friday API 请求失败: {e}")
            raise
        if "error" in data:
            err = data["error"]
            raise RuntimeError(f"Friday API 错误: {err.get('message', err)}")
        msg = data["choices"][0]["message"]
        # glm-5.2 是思考模型: reasoning_content=思考过程, content=最终答案
        # 优先返回最终答案 content; 为空(思考未结束)时退回 reasoning_content
        content = msg.get("content", "") or msg.get("reasoning_content", "")
        finish = data["choices"][0].get("finish_reason", "")
        if finish == "length" and not msg.get("content"):
            logger.warning(f"Friday[{model}] 思考未完成(max_tokens不足), 仅返回推理片段")
        usage = data.get("usage", {})
        logger.debug(f"Friday[{model}] tokens: {usage.get('total_tokens')}, finish={finish}")
        return content.strip()

    @staticmethod
    def _resolve_image(image: str, max_size: int = 1024) -> str:
        """本地路径 -> 压缩后的 data URI; URL 原样返回。

        max_size: 长边最大像素, 避免图片过大触发 413 Payload Too Large。
        """
        if image.startswith(("http://", "https://")):
            return image
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"图像不存在: {image}")
        # 压缩: 长边缩到 max_size, 转 JPEG 降体积
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            logger.debug(f"图像压缩: {path.name} {w}x{h} -> {img.size}, {len(buf.getvalue())//1024}KB")
            return f"data:image/jpeg;base64,{b64}"
        except ImportError:
            # 无 PIL, 回退原始 base64
            b64 = base64.b64encode(path.read_bytes()).decode()
            ext = path.suffix.lstrip(".").lower() or "png"
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64,{b64}"


def quick_test(app_id: str) -> bool:
    """快速测试 AppID 和模型可用性。"""
    fc = FridayClient(app_id)
    try:
        r = fc.chat("回复OK", model="glm-5.2", max_tokens=30)
        logger.info(f"glm-5.2 文本: {r[:40]}")
        print(f"[OK] glm-5.2 可用: {r[:40]}")
        return True
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"[FAIL] {e}")
        return False
