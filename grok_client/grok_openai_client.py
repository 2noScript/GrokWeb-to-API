from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import Any, overload

from .client import GrokClient

logger = logging.getLogger(__name__)


class _StreamWrapper:
    """Yields OpenAI-format chunk dicts while streaming from GrokClient."""

    def __init__(
            self,
            client: GrokClient,
            prompt: str,
            model_name: str,
            temperature: float | None = None,
            max_tokens: int | None = None,
    ):
        self._client = client
        self._prompt = prompt
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens

    def __iter__(self) -> Generator[dict[str, Any], None, None]:
        import time
        full = self._client.send_message(
            self._prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        created = int(time.time())
        base = {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": self._model_name,
        }

        for token in full.split():
            yield {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token + " "},
                        "finish_reason": None,
                    }
                ],
            }

        yield {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }


class GrokOpenAIClient:
    """
    OpenAI-compatible wrapper around the raw GrokClient.

    Lets you use the same interface as openai.OpenAI.chat.completions.create
    but talks directly to grok.com via GrokClient, without needing the
    FastAPI server.
    """

    def __init__(
            self,
            cookies: dict[str, str] | None = None,
            model_name: str = "grok-3",
    ):
        self.model_name = model_name
        self._client = GrokClient(cookies if cookies is not None else {})

    @overload
    def chat_completion(
            self,
            messages: list[dict[str, str]],
            stream: bool = True,
            temperature: float = 1.0,
            max_tokens: int | None = None,
            response_format: dict[str, str] | None = None,
            functions: list[dict[str, Any]] | None = None,
            function_call: str | dict[str, str] | None = None,
    ) -> _StreamWrapper:
        ...

    @overload
    def chat_completion(
            self,
            messages: list[dict[str, str]],
            stream: bool = False,
            temperature: float = 1.0,
            max_tokens: int | None = None,
            response_format: dict[str, str] | None = None,
            functions: list[dict[str, Any]] | None = None,
            function_call: str | dict[str, str] | None = None,
    ) -> dict[str, Any]:
        ...

    def chat_completion(
            self,
            messages: list[dict[str, str]],
            stream: bool = False,
            temperature: float = 1.0,
            max_tokens: int | None = None,
            response_format: dict[str, str] | None = None,
            functions: list[dict[str, Any]] | None = None,
            function_call: str | dict[str, str] | None = None,
    ) -> _StreamWrapper | dict[str, Any]:
        prompt = self._messages_to_prompt(
            messages, response_format, functions, function_call
        )
        if stream:
            return _StreamWrapper(
                self._client,
                prompt,
                self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raw = self._client.send_message(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._wrap_response(raw)

    @staticmethod
    def process_streaming_response(stream: _StreamWrapper) -> str:
        full = ""
        for chunk in stream:
            choices: list[dict[str, Any]] = chunk.get("choices", [{}])
            delta: dict[str, Any] = choices[0].get("delta", {})
            content: str = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                full += content
        print()
        return full

    @staticmethod
    def _messages_to_prompt(
            messages: list[dict[str, str]],
            response_format: dict[str, str] | None = None,
            functions: list[dict[str, Any]] | None = None,
            function_call: str | dict[str, str] | None = None,
    ) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")

        if response_format and response_format.get("type") == "json_object":
            parts.insert(
                0,
                "system: You are a helpful assistant that always responds in valid JSON format.",
            )

        if functions:
            if function_call:
                if isinstance(function_call, dict):
                    func_name = function_call.get("name", "auto")
                else:
                    func_name = function_call
                parts.insert(
                    0,
                    f"system: You must call the function '{func_name}' with appropriate arguments. "
                    f"Available functions: {json.dumps([f.get('name') for f in functions])}. "
                    f"Schemas: {json.dumps(functions)}",
                )
            else:
                parts.insert(
                    0,
                    f"system: Available functions: {json.dumps([f.get('name') for f in functions])}. "
                    f"Schemas: {json.dumps(functions)}",
                )

        return "\n".join(parts)

    def _wrap_response(self, raw: str) -> dict[str, Any]:
        import time
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": raw,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
