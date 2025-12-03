from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional


class MissingDependencyError(RuntimeError):
    """Raised when an optional provider dependency is unavailable."""


class MissingCredentialError(RuntimeError):
    """Raised when the required API token/key is not configured."""


Provider = Literal["llama", "gemini"]


@dataclass
class LLMConfig:
    provider: Provider = "llama"
    model: Optional[str] = None
    temperature: float = 0.2
    max_output_tokens: int = 4000
    timeout: float = 30.0
    retries: int = 3
    retry_backoff: float = 2.0


class LLMClient:
    """
    Thin wrapper around either Hugging Face (Meta Llama) or Gemini APIs.

    The class is intentionally minimal so it can be mocked during tests.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._provider = self.config.provider
        if self._provider not in ("llama", "gemini"):
            raise ValueError(f"Unsupported provider: {self._provider}")

        if self._provider == "llama":
            self._setup_llama_client()
        else:
            self._setup_gemini_client()

    def _setup_llama_client(self) -> None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise MissingCredentialError(
                "HF_TOKEN environment variable is required for Llama provider."
            )
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise MissingDependencyError(
                "huggingface-hub is required for the Llama provider."
            ) from exc

        model = self.config.model or "mistralai/Mistral-7B-Instruct-v0.2"
        self._client = InferenceClient(
            model=model,
            token=token,
        )

    def _setup_gemini_client(self) -> None:
        # Read from env var, or use hardcoded fallback
        api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyC1hmISVTGqwXMQ7RodD6B9er9gf85vM0Q"
        if not api_key:
            raise MissingCredentialError(
                "GEMINI_API_KEY environment variable is required for Gemini provider."
            )

        import requests as _requests
        self._gemini_api_key = api_key
        # Use the thinking model again
        self._gemini_model = self.config.model or "gemini-2.5-flash"
        self._requests = _requests


    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Execute a chat completion request with retries.

        Returns the raw text output.
        """
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt < self.config.retries:
            try:
                if self._provider == "llama":
                    return self._call_llama(system_prompt, user_prompt)
                return self._call_gemini(system_prompt, user_prompt)
            except Exception as err:  # pylint: disable=broad-except
                last_err = err
                attempt += 1
                if attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_backoff ** attempt)
        raise RuntimeError(f"LLM request failed after retries: {last_err}") from last_err
        
    def _call_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call Gemini 2.5 Flash with a *capped* thinking budget.
        We explicitly limit thinking so the model can't eat the
        entire token budget on thoughts and return no answer.
        """

        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self._gemini_model}:generateContent?key={self._gemini_api_key}"
        )

        thinking_budget = 4000

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": system_prompt + "\n\n" + user_prompt
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "thinkingConfig": {
                    "thinkingBudget": thinking_budget
                },
            },
        }

        resp = self._requests.post(url, json=payload, timeout=self.config.timeout)

        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Gemini HTTP error {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()

        # ---- Pull the first non-thought text part, if any ----
        try:
            candidates = data["candidates"]
            if not candidates:
                raise KeyError("no candidates")

            content = candidates[0]["content"]
            parts = content.get("parts", [])

            for part in parts:
                txt = part.get("text")
                if not txt:
                    continue
                # If Google ever sets part["thought"] for summaries, skip them
                if part.get("thought"):
                    continue
                return txt

        except (KeyError, IndexError, TypeError):
            # fall through to error below
            pass

        usage = data.get("usageMetadata", {})
        raise RuntimeError(
            "Gemini response contained no usable text.\n"
            f"finishReason={data.get('candidates',[{}])[0].get('finishReason')}, "
            f"promptTokenCount={usage.get('promptTokenCount')}, "
            f"thoughtsTokenCount={usage.get('thoughtsTokenCount')}, "
            f"totalTokenCount={usage.get('totalTokenCount')}.\n"
            f"Raw response: {data}"
        )

