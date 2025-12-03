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
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise MissingCredentialError(
                "GEMINI_API_KEY environment variable is required for Gemini provider."
            )

        try:
            import requests as _requests
        except ImportError as exc:
            raise MissingDependencyError(
                "requests is required for the Gemini provider."
            ) from exc

        self._gemini_api_key = api_key
        # Use the thinking model again
        self._gemini_model = self.config.model or "gemini-2.5-flash"
        self._requests = _requests
        self._session = _requests.Session()


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

    def _call_llama(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Hugging Face Inference API for chat-style completion."""

        prompt = (system_prompt + "\n\n" + user_prompt).strip()
        response = self._client.text_generation(
            prompt,
            max_new_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            stop_sequences=["</s>"],
        )

        # `text_generation` returns a string for non-streaming usage
        if isinstance(response, str):
            return response.strip()

        # Some versions return an object with `generated_text`
        generated = getattr(response, "generated_text", None)
        if isinstance(generated, str):
            return generated.strip()

        raise RuntimeError(f"Unexpected Llama response type: {type(response)!r}")

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

        try:
            resp = self._session.post(url, json=payload, timeout=self.config.timeout)
            resp.raise_for_status()
        except self._requests.RequestException as exc:  # pragma: no cover - HTTP errors
            status = getattr(resp, "status_code", "unknown") if "resp" in locals() else "unknown"
            error_summary = self._summarize_gemini_error(resp) if "resp" in locals() else "<no response>"
            raise RuntimeError(f"Gemini HTTP error {status}: {error_summary}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Gemini returned non-JSON response: {resp.text[:500]}"
            ) from exc

        # ---- Pull the first non-thought text part, if any ----
        text = self._extract_gemini_text(data)
        if text:
            return text

        usage = data.get("usageMetadata", {})
        raise RuntimeError(
            "Gemini response contained no usable text.\n"
            f"finishReason={data.get('candidates',[{}])[0].get('finishReason')}, "
            f"promptTokenCount={usage.get('promptTokenCount')}, "
            f"thoughtsTokenCount={usage.get('thoughtsTokenCount')}, "
            f"totalTokenCount={usage.get('totalTokenCount')}.\n"
            f"Raw response: {data}"
        )

    def _summarize_gemini_error(self, resp) -> str:
        """Return a concise error description with credential guidance when relevant."""

        suggestion = ""
        if resp.status_code in (401, 403):
            suggestion = " (check GEMINI_API_KEY; rotate if revoked or reported leaked)"

        try:
            payload = resp.json()
            error_info = payload.get("error") if isinstance(payload, dict) else None
        except ValueError:
            error_info = None

        if isinstance(error_info, dict):
            code = error_info.get("code")
            status = error_info.get("status")
            message = error_info.get("message")
            details = ", ".join(str(part) for part in (code, status, message) if part)
            if details:
                return details + suggestion

        return resp.text[:500] + suggestion

    @staticmethod
    def _extract_gemini_text(data: dict) -> Optional[str]:
        """Return the first non-thought text part from a Gemini response."""

        candidates = data.get("candidates")
        if not candidates or not isinstance(candidates, list):
            return None

        first_candidate = candidates[0] or {}
        content = first_candidate.get("content", {})
        parts = content.get("parts") if isinstance(content, dict) else None
        if not parts or not isinstance(parts, list):
            return None

        for part in parts:
            if not isinstance(part, dict):
                continue
            txt = part.get("text")
            if not txt or part.get("thought"):
                continue
            return txt

        return None

