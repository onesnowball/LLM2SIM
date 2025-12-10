from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


class MissingDependencyError(RuntimeError):
    """Raised when the optional OpenAI dependency is unavailable."""


class MissingCredentialError(RuntimeError):
    """Raised when the required OpenAI API key is not configured."""


Provider = Literal["openai"]


@dataclass
class LLMConfig:
    """
    Config for the LLM client (OpenAI only).

    - provider: always "openai".
    - model: OpenAI model name (gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o4-mini, gpt-5)
    - temperature: sampling temperature.
    - max_output_tokens: maximum tokens the model may generate.
    - timeout: total time budget per call (used in retry logic, not per-request).
    - retries: number of times to retry on failure.
    - retry_backoff: multiplicative factor for sleep between retries.
    - openai_api_key: optional explicit key; otherwise uses OPENAI_API_KEY env var.
    - response_format: optional OpenAI response_format dict
    """

    provider: Provider = "openai"
    model: Optional[str] = None
    temperature: float = 0.0  
    max_output_tokens: int = 16000  
    timeout: float = 60.0 
    retries: int = 3
    retry_backoff: float = 2.0

    openai_api_key: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None


class LLMClient:
    """
    Thin wrapper around the OpenAI Chat Completions API.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._provider: Provider = config.provider

        if self._provider != "openai":
            raise ValueError(f"Unsupported provider: {self._provider}. Only 'openai' is supported.")

        self._setup_openai_client()


    def _setup_openai_client(self) -> None:
        api_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MissingCredentialError(
                "OPENAI_API_KEY environment variable is required for the OpenAI provider."
            )

        try:
            from openai import OpenAI 
        except ImportError as exc:
            raise MissingDependencyError(
                "The 'openai' package is required. Install it with `pip install --upgrade openai`."
            ) from exc

        self._openai_client = OpenAI(api_key=api_key)
        
        model_map = {
            "gpt-4.1": "gpt-4.1",
            "gpt-4.1-mini": "gpt-4.1-mini", 
            "gpt-4.1-nano": "gpt-4.1-nano",
            "o4-mini": "o4-mini",
            "gpt-5": "gpt-5",
            "gpt-4o": "gpt-4.1",
            "gpt-4o-mini": "gpt-4.1-mini",
        }
        
        requested_model = self.config.model or "gpt-4.1"
        self._openai_model = model_map.get(requested_model, requested_model)


    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the model with a system + user prompt and return the text content.
        Retries on errors up to config.retries times with exponential backoff.
        """
        last_exc: Optional[Exception] = None
        start_time = time.time()

        for attempt in range(1, self.config.retries + 1):
            try:
                return self._call_openai(system_prompt, user_prompt)
            except Exception as exc:  
                last_exc = exc
                elapsed = time.time() - start_time
                if attempt >= self.config.retries or elapsed > self.config.timeout:
                    raise RuntimeError(
                        f"LLM request failed after {attempt} attempt(s): {exc}"
                    ) from exc

                sleep_sec = (self.config.retry_backoff ** (attempt - 1)) * 1.0
                print(f"  Retry {attempt}/{self.config.retries} after {sleep_sec:.1f}s...")
                time.sleep(sleep_sec)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed for an unknown reason.")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """
        Low-level call to OpenAI's chat.completions API.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {}
        if self.config.response_format is not None:
            kwargs["response_format"] = self.config.response_format

        if self._openai_model in ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "gpt-5"]:
            kwargs["max_completion_tokens"] = self.config.max_output_tokens

        else:
            kwargs["max_tokens"] = self.config.max_output_tokens
            kwargs["temperature"] = self.config.temperature

        resp = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
            **kwargs,
        )

        if not resp.choices:
            raise RuntimeError(f"OpenAI returned no choices: {resp!r}")

        message = resp.choices[0].message
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected OpenAI response format: {resp!r}")

        return content.strip()