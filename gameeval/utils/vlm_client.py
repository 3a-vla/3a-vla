"""Async client for OpenAI-compatible vision-language model endpoints."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import numpy as np

# Fields whose names indicate an HTTP header rather than a body field.
_REQUEST_ID_HEADERS = frozenset({"X-Request-Id", "X-Request-ID", "Request-Id"})


class VLMClient:
    """Small OpenAI-compatible VLM client shared by agents and evaluators.

    ``base_url`` may point at any server implementing the OpenAI chat
    completions API. Credentials come from ``api_key`` or ``OPENAI_API_KEY``.

    Some OpenAI-compatible deployments require deployment-specific request
    headers or body fields (routing identifiers, sampling controls, or switches
    such as disabling a reasoning mode). ``extra_headers`` and ``extra_body``
    forward those verbatim, while ``request_id_fields`` names the fields that
    must carry a unique per-request identifier. Set ``stream`` for deployments
    that only serve streamed completions; the deltas are joined before the
    caller sees them. ``empty_response_retries`` guards against deployments that
    occasionally return no content at all.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 30.0,
        image_detail: str = "low",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        request_id_fields: list[str] | tuple[str, ...] | None = None,
        stream: bool = False,
        empty_response_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.image_detail = image_detail
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.request_id_fields = tuple(request_id_fields or ())
        self.stream = bool(stream)
        self.empty_response_retries = max(0, int(empty_response_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._openai_client: Any = None

    async def chat(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        system_prompt: str | None = None,
        user_text: str | None = None,
        images_b64: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat-completions request and return assistant text.

        Some deployments intermittently close a stream having emitted no content.
        Treating that as a verdict would silently drop the episode from coverage,
        so an empty result is retried before being surfaced.
        """
        request_messages = messages or self._build_messages(
            system_prompt, user_text, images_b64
        )
        text = ""
        for attempt in range(self.empty_response_retries + 1):
            text = await self._call_openai(request_messages, temperature, max_tokens)
            if text.strip():
                return text
            if attempt < self.empty_response_retries:
                await asyncio.sleep(self.retry_backoff_seconds * (attempt + 1))
        return text

    def chat_sync(
        self,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous wrapper that also works when called from an event loop."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self.chat(messages, **kwargs)).result()
        except RuntimeError:
            return asyncio.run(self.chat(messages, **kwargs))

    def _build_messages(
        self,
        system_prompt: str | None,
        user_text: str | None,
        images_b64: list[str] | None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})

        user_content: list[dict[str, Any]] = []
        for image_b64 in images_b64 or []:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self.image_detail,
                    },
                }
            )
        user_content.append({"type": "text", "text": user_text or " "})
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        normalized = (model or "").lower().lstrip()
        return normalized.startswith(("gpt-5", "o1", "o3", "o4"))

    def _per_request_fields(self) -> tuple[dict[str, str], dict[str, Any]]:
        """Build headers and body fields, filling per-request correlation ids.

        Some deployments reject requests without a unique request identifier.
        Configuring a fixed one would make every call share an id, so any
        ``request_id_fields`` are generated fresh per request.
        """
        headers = dict(self.extra_headers)
        body = dict(self.extra_body)
        if not self.request_id_fields:
            return headers, body

        request_id = f"gameeval_{uuid.uuid4()}"
        for field in self.request_id_fields:
            if field in _REQUEST_ID_HEADERS:
                headers[field] = request_id
            else:
                body[field] = request_id
        return headers, body

    async def _call_openai(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        import openai

        if self._openai_client is None:
            client_args: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_args["base_url"] = self.base_url
            if self.timeout:
                client_args["timeout"] = self.timeout
            self._openai_client = openai.AsyncOpenAI(**client_args)

        effective_tokens = max_tokens if max_tokens is not None else self.max_tokens
        effective_temperature = (
            temperature if temperature is not None else self.temperature
        )
        call_args: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": effective_temperature,
        }
        token_key = (
            "max_completion_tokens"
            if self._uses_max_completion_tokens(self.model)
            else "max_tokens"
        )
        call_args[token_key] = effective_tokens

        headers, body = self._per_request_fields()
        if headers:
            call_args["extra_headers"] = headers
        if body:
            call_args["extra_body"] = body

        if self.stream:
            call_args["stream"] = True
            return await self._collect_stream(call_args)
        response = await self._openai_client.chat.completions.create(**call_args)
        return self._extract_text(response)

    async def _collect_stream(self, call_args: dict[str, Any]) -> str:
        """Concatenate a streamed completion into one response string.

        Some deployments only serve ``chat.completion.chunk`` events. The judge
        needs the whole verdict, so deltas are joined before parsing. Any
        ``reasoning_content`` is deliberately dropped: only the final answer is
        part of the judge protocol.
        """
        stream = await self._openai_client.chat.completions.create(**call_args)
        parts: list[str] = []
        async for chunk in stream:
            for choice in getattr(chunk, "choices", None) or []:
                content = getattr(getattr(choice, "delta", None), "content", None)
                if content:
                    parts.append(content)
        return "".join(parts)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if hasattr(response, "choices"):
            return response.choices[0].message.content or ""
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["message"]["content"] or ""
        return str(response)

    @staticmethod
    def encode_image(image: np.ndarray, fmt: str = "jpeg") -> str:
        """Encode an RGB numpy image as base64."""
        from gameeval.utils.screenshot import encode_image_base64

        return encode_image_base64(image, fmt=fmt)
