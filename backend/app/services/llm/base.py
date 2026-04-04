"""
LLM Provider Base Classes
=========================
Abstract interfaces for LLM text/vision generation and embedding.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

import numpy as np

from app.services.llm.types import LLMMessage, LLMResult, StreamChunk


class LLMProvider(ABC):
    """Abstract interface for LLM text/multimodal generation."""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """
        Synchronous text generation.

        Args:
            messages: Conversation history (may include images).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            system_prompt: System-level instruction (provider handles injection).
            think: If True and supported, return LLMResult with thinking text.

        Returns:
            Generated text string, or LLMResult when think=True.
        """
        ...

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """
        Async text generation.
        Default: runs complete() in a thread pool.
        Providers with native async support should override.
        """
        return await asyncio.to_thread(
            self.complete,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async streaming generation. Yields StreamChunk objects.

        Default fallback: calls acomplete() and yields a single text chunk.
        Providers with native streaming should override this method.
        """
        result = await self.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )
        if isinstance(result, LLMResult):
            if result.thinking:
                yield StreamChunk(type="thinking", text=result.thinking)
            yield StreamChunk(type="text", text=result.content)
        else:
            yield StreamChunk(type="text", text=result)

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider/model supports image inputs."""
        ...

    def supports_thinking(self) -> bool:
        """Whether this provider/model supports thinking mode."""
        return False

    def supports_native_tools(self) -> bool:
        """Whether this provider/model supports native tool calling."""
        return False


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding generation (used by KG)."""

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> np.ndarray:
        """
        Synchronous batch embedding.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        ...

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Async batch embedding.
        Default: runs embed_sync() in a thread pool.
        """
        return await asyncio.to_thread(self.embed_sync, texts)

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding vector dimension for this model."""
        ...
