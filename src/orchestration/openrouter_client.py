"""
OpenRouter chat client (OpenAI-compatible HTTP API).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from config import config

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Thin wrapper around OpenRouter's /chat/completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or config.llm.openrouter_api_key
        self.base_url = (base_url or config.llm.openrouter_base_url).rstrip("/")
        self.model = model or config.llm.model_name
        self.max_tokens = config.llm.max_tokens
        self.temperature = config.llm.temperature

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if config.llm.site_url:
            headers["HTTP-Referer"] = config.llm.site_url
        if config.llm.site_name:
            headers["X-OpenRouter-Title"] = config.llm.site_name
        return headers

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant text.

        Args:
            messages: OpenAI-style message list [{"role": "...", "content": "..."}]
            model: Override default model id
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Assistant message content as a string
        """
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }

        url = f"{self.base_url}/chat/completions"
        logger.info("OpenRouter request model=%s", payload["model"])

        response = requests.post(
            url=url,
            headers=self._headers(),
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data}") from exc
