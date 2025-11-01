#!/usr/bin/env python3
"""Helpers for configuring OpenAI model runners."""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from dotenv import load_dotenv
from openai import OpenAI

__all__ = ["OpenAIConfigError", "create_openai_runner"]


class OpenAIConfigError(RuntimeError):
    """Raised when OpenAI cannot be configured for use."""


def _resolve_api_key() -> str:
    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    raise OpenAIConfigError(
        "OPENAI_API_KEY is not set. Export it or store it in a .env file at the project root."
    )


def create_openai_runner(
    model: str = "gpt-5-mini",
    *,
    default_kwargs: Mapping[str, Any] | None = None,
) -> Callable[..., str]:
    api_key = _resolve_api_key()
    client = OpenAI(api_key=api_key)
    base_kwargs = dict(default_kwargs or {})

    def run(prompt: str, /, **kwargs: Any) -> str:
        params = base_kwargs.copy()
        params.update(kwargs)

        try:
            response = client.responses.create(model=model, input=prompt, **params)
        except Exception as exc:  # pragma: no cover - network failures
            raise OpenAIConfigError(f"OpenAI request failed: {exc}") from exc

        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text.strip() or text

        raise OpenAIConfigError("Unexpected response format: no textual output returned.")

    return run
