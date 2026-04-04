"""
Chat Agent — Semi-Agentic SSE Streaming for NexusRAG
====================================================

Provides an SSE streaming endpoint where the LLM decides whether to call
``search_documents`` or answer directly, streaming thinking + tokens in real-time.

SSE Event Types:
  - status:         {"step": str, "detail": str}
  - thinking:       {"text": str}
  - sources:        {"sources": [...]}
  - images:         {"image_refs": [...]}
  - token:          {"text": str}
  - token_rollback: {}
  - complete:       {"answer": str, "sources": [...], ...}
  - error:          {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
import uuid
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.models.knowledge_base import KnowledgeBase
from app.models.document import DocumentImage
from app.schemas.rag import (
    ChatRequest,
    ChatSourceChunk,
    ChatImageRef,
)
from app.services.llm.types import LLMMessage, LLMImagePart, StreamChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_ITERATIONS = 3
MAX_VISION_IMAGES = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds

_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID."""
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

# Gemini native function calling
def _get_gemini_tool():
    """Lazily create Gemini Tool to avoid import at module level."""
    from google.genai import types
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "Search the knowledge base for relevant document sections. "
                "Use this tool when the user asks about document content, data, or facts. "
                "IMPORTANT: Rewrite the user's question as a detailed, specific search query "
                "to get better retrieval results. "
                "Do NOT use this tool for greetings, chitchat, or non-document questions."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "A rewritten, detailed search query based on the user's question. "
                            "Examples: 'revenue?' → 'total revenue figures and financial performance metrics'. "
                            "'AI là gì?' → 'định nghĩa trí tuệ nhân tạo, lịch sử và ứng dụng'"
                        ),
                    },
                    "top_k": {
                        "type": "INTEGER",
                        "description": "Number of relevant chunks to retrieve (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
    ])



# ---------------------------------------------------------------------------
# Ollama prompt-based tool calling — MANDATORY search before answering
# ---------------------------------------------------------------------------

OLLAMA_TOOL_SYSTEM = """\
## TOOL: search_documents

You have ONE tool: search_documents.  You call it by outputting EXACTLY:

<tool_call>{"name": "search_documents", "arguments": {"query": "<rewritten query>"}}</tool_call>

### ABSOLUTE RULES (violations are FATAL errors)

1. **Except for simple conversational messages, ALWAYS CALL search_documents FIRST.**
   Simple conversational messages that do NOT require a tool call:
   - Greetings: "hello", "xin chào", "hi", "hey", "good morning", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", "got it", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
   For ALL other messages — questions, requests, factual queries, analysis — you MUST
   call search_documents before answering. Your knowledge is UNRELIABLE; only document
   sources are trustworthy. If you are unsure whether a message needs a search, SEARCH.

2. **Your ENTIRE first response to a searchable query must be ONLY the <tool_call> block.**
   No text before it. No text after it. No explanation. Just the tool call.

3. **NEVER say "the documents do not contain this information" UNLESS you have already
   called search_documents in THIS turn.** Even if previous context seems sufficient,
   you MUST search again — previous results may be incomplete.

4. **Rewrite the query** to be specific and detailed.
   "doanh thu" → "doanh thu thuần, tổng doanh thu theo năm, tăng trưởng doanh thu"
   "AI model" → "AI model architecture, performance benchmarks, training details"

5. After receiving search results, answer using ONLY those sources with citations.
   Format: claim text[source_id]. Example: Doanh thu đạt 4.850 tỷ VNĐ[a3x9].

6. **ALWAYS include relevant rich content from sources:**
   - Images/Diagrams/Charts: reference as [IMG-xxxx]
   - Tables: reproduce key data preserving structure
   - Math formulas: include LaTeX exactly as in sources
   - Code snippets: include with language tags
"""

OLLAMA_TOOL_REMINDER = (
    "\n\n[SYSTEM REMINDER] If this is a question or request, you MUST call search_documents FIRST. "
    "Output ONLY: <tool_call>{\"name\": \"search_documents\", \"arguments\": {\"query\": \"...\"}}</tool_call> "
    "Exception: simple greetings, thanks, or farewells do NOT require a tool call — respond directly. "
    "For everything else, searching is MANDATORY."
)

# ---------------------------------------------------------------------------
# Gemini system prompt reinforcement — enforce tool calling for questions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ollama native tool calling (Gemma 4, Qwen 3.5, Llama 4, etc.)
# ---------------------------------------------------------------------------

def _get_ollama_native_tool() -> list[dict]:
    """Tool definition in OpenAI-compatible format for Ollama native tool calling."""
    return [{
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the knowledge base for relevant document sections. "
                "Use this tool when the user asks about document content, data, or facts. "
                "IMPORTANT: Rewrite the user's question as a detailed, specific search query "
                "to get better retrieval results. "
                "Do NOT use this tool for greetings, chitchat, or non-document questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A rewritten, detailed search query based on the user's question. "
                            "Examples: 'revenue?' -> 'total revenue figures and financial performance metrics'. "
                            "'AI la gi?' -> 'dinh nghia tri tue nhan tao, lich su va ung dung'"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    }]


OLLAMA_NATIVE_TOOL_SYSTEM = """\

## Tool Usage (MANDATORY)

You have a tool called `search_documents` that searches the knowledge base.

### ABSOLUTE RULES:
1. For ALL user questions, requests, factual queries, or analysis — you MUST call \
`search_documents` FIRST before answering. Your knowledge is UNRELIABLE; only \
document sources are trustworthy.
2. Only skip the tool call for simple conversational messages:
   - Greetings: "hello", "xin chào", "hi", "hey", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
3. NEVER say "the documents do not contain this information" or similar UNLESS you \
have already called `search_documents` in THIS turn. Even if previous context \
seems sufficient, you MUST search again — previous results may be incomplete.
4. Rewrite the user's query to be specific and detailed for better retrieval.
5. After receiving search results, answer using ONLY those sources with citations.
   Format: claim text[source_id]. Example: Doanh thu đạt 4.850 tỷ VNĐ[a3x9].
6. ALWAYS include relevant rich content from sources in your answer:
   - **Images/Diagrams/Charts**: Reference as [IMG-xxxx] when sources mention them.
   - **Tables**: Reproduce key data from tables, preserving structure.
   - **Math formulas**: Include LaTeX formulas exactly as they appear in sources.
   - **Code snippets**: Include code blocks with language tags.
   Do NOT omit these — they are essential parts of the answer.
"""

GEMINI_TOOL_SYSTEM = """\

## Tool Usage (MANDATORY)

You have a tool called `search_documents` that searches the knowledge base.

### ABSOLUTE RULES:
1. For ALL user questions, requests, factual queries, or analysis — you MUST call \
`search_documents` FIRST before answering. Even if the conversation history \
contains relevant information, you MUST search again to get fresh, accurate sources.
2. Only skip the tool call for simple conversational messages:
   - Greetings: "hello", "xin chào", "hi", "hey", etc.
   - Acknowledgements: "cảm ơn", "thank you", "thanks", "ok", etc.
   - Farewells: "bye", "goodbye", "tạm biệt", etc.
3. NEVER answer a question using information from previous turns without searching. \
Your previous answers may contain outdated or incomplete information.
4. NEVER say "the documents do not contain this information" or similar UNLESS you \
have already called `search_documents` in THIS turn.
5. NEVER reuse citation IDs from previous answers. Each answer must have its own \
fresh sources from a new search.
6. Rewrite the user's query to be specific and detailed for better retrieval.
7. ALWAYS include relevant rich content from sources in your answer:
   - **Images/Diagrams/Charts**: Reference as [IMG-xxxx] when sources mention them.
   - **Tables**: Reproduce key data from tables, preserving structure.
   - **Math formulas**: Include LaTeX formulas exactly as they appear in sources.
   - **Code snippets**: Include code blocks with language tags.
   Do NOT omit these — they are essential parts of the answer.
"""


# ---------------------------------------------------------------------------
# SSE Helpers (ported from PageIndex backend/app/api/v1/chat.py)
# ---------------------------------------------------------------------------

def format_sse_event(event: str, data: dict) -> str:
    """Format data as an SSE event string."""
    json_data = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


async def sse_with_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with periodic heartbeat comments.

    SSE spec allows lines starting with ':' as comments — browsers/clients
    silently ignore them but they keep the TCP connection alive, preventing
    timeouts when the upstream LLM takes a long time to respond.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _pump():
        try:
            async for event in source:
                await queue.put(event)
        except Exception:
            pass
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Tool executor — retrieval via NexusRAG
# ---------------------------------------------------------------------------

async def _execute_search_documents(
    workspace_id: int,
    query: str,
    top_k: int,
    db: AsyncSession,
    existing_ids: set[str],
) -> tuple[str, list[ChatSourceChunk], list[ChatImageRef], list[dict]]:
    """Execute document search and return formatted context + structured sources.

    Returns:
        (context_text, sources, image_refs, image_parts_for_vision)
    """
    from app.services.rag_service import get_rag_service
    from app.services.nexus_rag_service import NexusRAGService
    from pathlib import Path as _P
    from app.core.config import settings

    rag_service = get_rag_service(db, workspace_id)

    chunks = []
    citations = []
    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=query,
            top_k=min(top_k, 10),
            mode="hybrid",
            include_images=False,
        )
        chunks = result.chunks
        citations = result.citations
    else:
        from types import SimpleNamespace
        legacy = rag_service.query(question=query, top_k=min(top_k, 10))
        for i, c in enumerate(legacy.chunks):
            chunks.append(SimpleNamespace(
                content=c.content,
                document_id=int(c.metadata.get("document_id", 0)),
                chunk_index=i,
                page_no=int(c.metadata.get("page_no", 0)),
                heading_path=str(c.metadata.get("heading_path", "")).split(" > ") if c.metadata.get("heading_path") else [],
                source_file=str(c.metadata.get("source", "")),
                image_refs=[],
            ))

    # Build sources
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(existing_ids)
        existing_ids.add(cid)
        sources.append(ChatSourceChunk(
            index=cid,
            chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
            content=chunk.content,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            score=0.0,
            source_type="vector",
        ))
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    context = "\n\n---\n\n".join(context_parts)

    # Build image references
    seen_image_ids: set[str] = set()
    chunk_image_ids: list[str] = []
    for c in chunks:
        for iid in getattr(c, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)

    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    if not resolved_images:
        source_pages = {
            (getattr(c, "document_id", 0), getattr(c, "page_no", 0))
            for c in chunks if getattr(c, "page_no", 0) > 0
        }
        if source_pages:
            from sqlalchemy import or_, and_
            page_filters = [
                and_(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
                for doc_id, page_no in source_pages
            ]
            img_result = await db.execute(
                select(DocumentImage).where(or_(*page_filters))
            )
            resolved_images = list(img_result.scalars().all())
            seen = set()
            deduped = []
            for img in resolved_images:
                if img.image_id not in seen:
                    seen.add(img.image_id)
                    deduped.append(img)
            resolved_images = deduped

    chat_image_refs: list[ChatImageRef] = []
    image_context_parts: list[str] = []
    image_parts: list[dict] = []

    for img in resolved_images[:MAX_VISION_IMAGES]:
        img_ref_id = _generate_citation_id(existing_ids)
        existing_ids.add(img_ref_id)
        img_url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
        chat_image_refs.append(ChatImageRef(
            ref_id=img_ref_id,
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            url=img_url,
            width=img.width,
            height=img.height,
        ))
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(f"- [IMG-{img_ref_id}] Page {img.page_no}: {cap}")

        img_path = _P(img.file_path)
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                mime = img.mime_type or "image/png"
                image_parts.append({
                    "inline_data": {"mime_type": mime, "data": img_bytes},
                    "page_no": img.page_no,
                    "caption": img.caption or "",
                    "img_ref_id": img_ref_id,
                })
            except Exception as e:
                logger.warning(f"Failed to read image {img.image_id}: {e}")

    if image_context_parts:
        context += "\n\nDocument Images:\n" + "\n".join(image_context_parts)

    return context, sources, chat_image_refs, image_parts


# ---------------------------------------------------------------------------
# Agent loop — semi-agentic streaming
# ---------------------------------------------------------------------------

async def agent_chat_stream(
    workspace_id: int,
    message: str,
    history: list[dict],
    enable_thinking: bool,
    db: AsyncSession,
    system_prompt: str,
    force_search: bool = False,
) -> AsyncGenerator[dict, None]:
    """Semi-agentic chat loop with streaming.

    - force_search=True: pre-search before calling LLM, inject sources as context.
      Guarantees retrieval for every query regardless of model tool-calling ability.
    - force_search=False (default): agentic tool-calling loop.
      Gemini uses native function calling; Ollama uses prompt-based tool calling.

    Yields dicts with 'event' and 'data' keys for SSE formatting.
    """
    from app.services.llm import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider()
    provider_name = settings.LLM_PROVIDER.lower()
    is_gemini = provider_name == "gemini"

    existing_ids: set[str] = set()
    all_sources: list[ChatSourceChunk] = []
    all_images: list[ChatImageRef] = []
    all_image_parts: list[dict] = []

    # Build conversation messages
    messages: list[LLMMessage] = []
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg["content"]))

    # Build user message
    messages.append(LLMMessage(role="user", content=message))

    # Tool / prompt setup
    tools = None
    effective_system_prompt = system_prompt
    is_ollama_native = False  # track for tool result handling

    if force_search:
        # ── Force-search mode: pre-search before LLM call ──────────────────
        # Retrieve sources immediately, inject as context. No tool calling needed.
        yield {"event": "status", "data": {"step": "retrieving", "detail": f"Searching: {message[:80]}..."}}

        context, sources, images, img_parts = await _execute_search_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)

        if sources:
            yield {"event": "sources", "data": {"sources": [s.model_dump() for s in sources]}}
        if images:
            yield {"event": "images", "data": {"image_refs": [i.model_dump() for i in images]}}

        if sources:
            tool_result_parts = [
                "I have retrieved the following document sources for you.\n",
                "=== DOCUMENT SOURCES ===",
                context,
                "=== END SOURCES ===\n",
                "IMPORTANT:\n"
                "- Read EVERY source above carefully. Answers often require "
                "combining data from MULTIPLE sources.\n"
                "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
                "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023.\n"
                "- If no source contains relevant information, say: "
                "\"Tài liệu không chứa thông tin này.\"\n",
            ]
            tool_result_content = "\n".join(tool_result_parts)

            user_images_fs: list[LLMImagePart] = []
            if img_parts:
                for img_data in img_parts:
                    tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
                    user_images_fs.append(LLMImagePart(
                        data=img_data["inline_data"]["data"],
                        mime_type=img_data["inline_data"]["mime_type"],
                    ))

            tool_result_content += f"\n\nNow answer the question: {message}"
            messages.append(LLMMessage(
                role="user",
                content=tool_result_content,
                images=user_images_fs,
            ))
        # tools remain None — model answers directly with provided context
    elif is_gemini:
        tools = [_get_gemini_tool()]
        # Reinforce tool-calling obligation in system prompt for Gemini
        effective_system_prompt = system_prompt + GEMINI_TOOL_SYSTEM
    elif provider.supports_native_tools():
        # Ollama with native tool calling (Gemma 4, Qwen 3.5, Llama 4, etc.)
        is_ollama_native = True
        tools = _get_ollama_native_tool()
        effective_system_prompt = system_prompt + "\n\n" + OLLAMA_NATIVE_TOOL_SYSTEM
    else:
        # Ollama prompt-based fallback (older models without native tool support)
        effective_system_prompt = system_prompt + "\n\n" + OLLAMA_TOOL_SYSTEM
        # Also append a reminder directly to the user message so the model
        # sees it right before generating — reinforces the tool requirement
        messages[-1] = LLMMessage(
            role="user",
            content=messages[-1].content + OLLAMA_TOOL_REMINDER,
        )

    yield {"event": "status", "data": {"step": "analyzing", "detail": "Analyzing your question..."}}

    accumulated_text = ""
    thinking_text = ""

    for iteration in range(MAX_AGENT_ITERATIONS):
        iteration_text = ""
        function_calls: list[dict] = []
        tokens_yielded = False

        async for chunk in provider.astream(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system_prompt,
            think=enable_thinking,
            tools=tools if (is_gemini or is_ollama_native) else None,
        ):
            if chunk.type == "thinking":
                thinking_text += chunk.text
                yield {"event": "thinking", "data": {"text": chunk.text}}
            elif chunk.type == "function_call":
                function_calls.append(chunk.function_call)
            elif chunk.type == "text":
                iteration_text += chunk.text
                # Speculative streaming — send tokens if no tool call seen yet
                if not function_calls:
                    accumulated_text += chunk.text
                    tokens_yielded = True
                    yield {"event": "token", "data": {"text": chunk.text}}

        if function_calls:
            # Rollback speculative tokens
            if tokens_yielded:
                accumulated_text = ""
                yield {"event": "token_rollback", "data": {}}

            fc = function_calls[0]
            fc_name = fc.get("name", "")
            fc_args = fc.get("args", {})

            if fc_name == "search_documents":
                query = fc_args.get("query", message)
                top_k = int(fc_args.get("top_k", 8))

                yield {"event": "status", "data": {
                    "step": "retrieving",
                    "detail": f"Searching: {query[:80]}..."
                }}

                context, sources, images, img_parts = await _execute_search_documents(
                    workspace_id, query, top_k, db, existing_ids,
                )
                all_sources.extend(sources)
                all_images.extend(images)
                all_image_parts.extend(img_parts)

                if sources:
                    yield {"event": "sources", "data": {
                        "sources": [s.model_dump() for s in sources]
                    }}
                if images:
                    yield {"event": "images", "data": {
                        "image_refs": [i.model_dump() for i in images]
                    }}

                # Build tool result as user message with sources
                tool_result_parts = [
                    "I have retrieved the following document sources for you.\n",
                    "=== DOCUMENT SOURCES ===",
                    context,
                    "=== END SOURCES ===\n",
                    "IMPORTANT:\n"
                    "- Read EVERY source above carefully. Answers often require "
                    "combining data from MULTIPLE sources.\n"
                    "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
                    "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023.\n"
                    "- RICH CONTENT: Include ALL relevant images [IMG-xxxx], tables, "
                    "math formulas (LaTeX), diagrams, and code snippets from sources. "
                    "These are essential — do NOT omit them.\n"
                    "- If no source contains relevant information, say: "
                    "\"Tài liệu không chứa thông tin này.\"\n",
                ]
                tool_result_content = "\n".join(tool_result_parts)

                # Add image inline references for vision models
                user_images: list[LLMImagePart] = []
                if img_parts:
                    for img_data in img_parts:
                        tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
                        user_images.append(LLMImagePart(
                            data=img_data["inline_data"]["data"],
                            mime_type=img_data["inline_data"]["mime_type"],
                        ))

                tool_result_content += f"\n\nNow answer the question: {message}"

                if is_gemini:
                    # Gemini: use native Content with thought_signature
                    # (required by Gemini 3 for proper multi-turn reasoning)
                    # and native FunctionResponse for the tool result.
                    from google.genai import types as _gtypes

                    raw_content = getattr(provider, "last_response_content", None)
                    if raw_content:
                        # Preserve the model's raw response (with thought_signature)
                        messages.append(LLMMessage(
                            role="assistant",
                            content="",
                            _raw_provider_content=raw_content,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=f"[Called search_documents(query=\"{query}\")]",
                        ))

                    # Build native FunctionResponse with sources context
                    func_resp_parts = [_gtypes.Part.from_function_response(
                        name="search_documents",
                        response={"result": tool_result_content},
                    )]
                    func_resp_content = _gtypes.Content(
                        role="user",
                        parts=func_resp_parts,
                    )
                    messages.append(LLMMessage(
                        role="user",
                        content="",
                        _raw_provider_content=func_resp_content,
                    ))

                    # Send images as a separate user message for vision
                    if img_parts:
                        img_llm_parts: list[LLMImagePart] = []
                        img_text = "Referenced document images:\n"
                        for img_data in img_parts:
                            img_text += f"[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']})\n"
                            img_llm_parts.append(LLMImagePart(
                                data=img_data["inline_data"]["data"],
                                mime_type=img_data["inline_data"]["mime_type"],
                            ))
                        messages.append(LLMMessage(
                            role="user",
                            content=img_text,
                            images=img_llm_parts,
                        ))

                    # Remove tool-calling instructions since search is done;
                    # keep tools so thinking + tool awareness still works.
                    effective_system_prompt = system_prompt
                elif is_ollama_native:
                    # Ollama native: preserve raw assistant tool_call response
                    # and send tool result via native "tool" role message.
                    raw_msg = getattr(provider, "last_response_message", None)
                    if raw_msg:
                        messages.append(LLMMessage(
                            role="assistant",
                            content="",
                            _raw_provider_content=raw_msg,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=f"[Called search_documents(query=\"{query}\")]",
                        ))

                    # Send tool result using Ollama's native tool message format
                    messages.append(LLMMessage(
                        role="tool",
                        content="",
                        _raw_provider_content={
                            "role": "tool",
                            "content": tool_result_content,
                        },
                    ))

                    # Send images as a separate user message for vision
                    if img_parts:
                        img_llm_parts: list[LLMImagePart] = []
                        img_text = "Referenced document images:\n"
                        for img_data in img_parts:
                            img_text += f"[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']})\n"
                            img_llm_parts.append(LLMImagePart(
                                data=img_data["inline_data"]["data"],
                                mime_type=img_data["inline_data"]["mime_type"],
                            ))
                        messages.append(LLMMessage(
                            role="user",
                            content=img_text,
                            images=img_llm_parts,
                        ))

                    # Remove tool-calling instructions; keep tools for awareness.
                    effective_system_prompt = system_prompt
                else:
                    # Ollama prompt-based: add text-based assistant + user messages
                    # to maintain proper user/assistant alternation
                    # (prevents two consecutive user messages which confuses
                    # small models like qwen3.5).
                    messages.append(LLMMessage(
                        role="assistant",
                        content=f"[Called search_documents(query=\"{query}\")]",
                    ))
                    messages.append(LLMMessage(
                        role="user",
                        content=tool_result_content,
                        images=user_images,
                    ))
                    # Remove tool prompt from system prompt so the model
                    # answers with sources instead of calling the tool again.
                    effective_system_prompt = system_prompt

                yield {"event": "status", "data": {
                    "step": "generating",
                    "detail": "Generating answer..."
                }}
            else:
                # Unknown tool — treat accumulated text as answer
                logger.warning(f"Unknown tool call: {fc_name}")
                break
        else:
            # No tool call from model — answer is in accumulated_text, done.
            break

    # ── Fallback: model produced no text and no search was done ──────────
    # Small Ollama models (e.g. qwen3.5:4b) may output thinking about
    # needing to search but never produce a <tool_call> tag or any text.
    # Auto-search and retry once to avoid "Unable to generate a response."
    if not accumulated_text and not all_sources and not is_gemini:
        logger.warning(
            "Ollama produced no text and no tool call — fallback to auto-search"
        )
        yield {"event": "status", "data": {
            "step": "retrieving",
            "detail": f"Searching: {message[:80]}..."
        }}

        context, sources, images, img_parts = await _execute_search_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)

        if sources:
            yield {"event": "sources", "data": {
                "sources": [s.model_dump() for s in sources]
            }}
        if images:
            yield {"event": "images", "data": {
                "image_refs": [i.model_dump() for i in images]
            }}

        if sources:
            fallback_parts = [
                "I have retrieved the following document sources for you.\n",
                "=== DOCUMENT SOURCES ===",
                context,
                "=== END SOURCES ===\n",
                "IMPORTANT:\n"
                "- Read EVERY source above carefully.\n"
                "- Include ALL relevant images [IMG-xxxx], tables, math formulas, "
                "and code snippets from sources.\n"
                "- If no source contains relevant information, say: "
                "\"Tài liệu không chứa thông tin này.\"\n",
            ]
            fallback_content = "\n".join(fallback_parts)
            fallback_content += f"\n\nNow answer the question: {message}"

            # Build CLEAN messages for retry — only keep recent history
            # and the user question with sources.  Drop the tool-calling
            # prompt that already failed, so the model focuses on answering.
            fallback_msgs: list[LLMMessage] = []
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "assistant"
                fallback_msgs.append(LLMMessage(role=role, content=msg["content"]))
            fallback_msgs.append(LLMMessage(role="user", content=fallback_content))

            yield {"event": "status", "data": {
                "step": "generating", "detail": "Generating answer..."
            }}

            async for chunk in provider.astream(
                fallback_msgs,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                system_prompt=system_prompt,  # original prompt without tool instructions
                think=enable_thinking,
                tools=None,
            ):
                if chunk.type == "thinking":
                    thinking_text += chunk.text
                    yield {"event": "thinking", "data": {"text": chunk.text}}
                elif chunk.type == "text":
                    accumulated_text += chunk.text
                    yield {"event": "token", "data": {"text": chunk.text}}

    # Extract related entities from KG (best-effort)
    related_entities: list[str] = []
    try:
        from app.api.rag import _get_kg_service
        kg = await _get_kg_service(workspace_id)
        entities = await kg.get_entities(limit=200)
        entity_names = {e["name"].lower(): e["name"] for e in entities}
        text_lower = accumulated_text.lower()
        for lower_name, original_name in entity_names.items():
            if len(lower_name) >= 2 and lower_name in text_lower:
                related_entities.append(original_name)
    except Exception:
        pass

    # Strip artifacts
    if accumulated_text:
        accumulated_text = re.sub(r'<unused\d+>:?\s*', '', accumulated_text).strip()

    yield {"event": "complete", "data": {
        "answer": accumulated_text or "Unable to generate a response.",
        "sources": [s.model_dump() for s in all_sources],
        "image_refs": [i.model_dump() for i in all_images],
        "thinking": thinking_text or None,
        "related_entities": related_entities[:30],
    }}


# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------

async def chat_stream_endpoint(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession,
):
    """SSE streaming chat endpoint.

    Called from rag.py router — not a standalone router to avoid circular imports.
    """
    # Verify workspace
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    # Build system prompt
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
    system_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT

    # Build history
    history = [{"role": m.role, "content": m.content} for m in request.history]

    # Persist user message immediately
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        user_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
        )
        db.add(user_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist user message: {e}")
        await db.rollback()

    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        final_sources = []
        final_images = []
        final_thinking = None
        final_entities = []

        # Collect agent steps for persistence (ThinkingTimeline survives reload)
        collected_steps: list[dict] = []
        step_counter = 0
        # Track sources/images as they arrive so sources_found inserts BEFORE generating
        streaming_sources: list[dict] = []
        streaming_images: list[dict] = []

        try:
            async for event in agent_chat_stream(
                workspace_id=workspace_id,
                message=request.message,
                history=history,
                enable_thinking=request.enable_thinking,
                db=db,
                system_prompt=system_prompt,
                force_search=request.force_search,
            ):
                event_type = event["event"]
                event_data = event["data"]

                # Collect status steps; insert sources_found before "generating"
                if event_type == "status":
                    step_name = event_data.get("step", "analyzing")

                    # When generating starts, insert sources_found first (correct order)
                    if step_name == "generating" and streaming_sources:
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })
                        streaming_sources.clear()
                        streaming_images.clear()

                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": step_name,
                        "detail": event_data.get("detail", ""),
                        "status": "completed",
                        "timestamp": 0,
                    })

                # Track sources/images as they arrive
                elif event_type == "sources":
                    streaming_sources.extend(event_data.get("sources", []))

                elif event_type == "images":
                    streaming_images.extend(event_data.get("image_refs", []))

                # Attach thinking text to the analyzing step
                elif event_type == "thinking":
                    thinking_fragment = event_data.get("text", "")
                    for s in collected_steps:
                        if s["step"] == "analyzing":
                            s["thinkingText"] = (s.get("thinkingText") or "") + thinking_fragment
                            break

                elif event_type == "complete":
                    final_answer = event_data.get("answer", "")
                    final_sources = event_data.get("sources", [])
                    final_images = event_data.get("image_refs", [])
                    final_thinking = event_data.get("thinking")
                    final_entities = event_data.get("related_entities", [])

                    # Fallback: if sources arrived but generating step was never emitted
                    if streaming_sources and not any(s["step"] == "sources_found" for s in collected_steps):
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })

                    # Done step
                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": "done",
                        "detail": "Done",
                        "status": "completed",
                        "timestamp": 0,
                    })

                yield format_sse_event(event_type, event_data)

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e)})
        finally:
            # Persist assistant message
            if final_answer:
                try:
                    from app.models.chat_message import ChatMessage as ChatMessageModel
                    assistant_row = ChatMessageModel(
                        workspace_id=workspace_id,
                        message_id=str(uuid.uuid4()),
                        role="assistant",
                        content=final_answer,
                        sources=final_sources if final_sources else None,
                        related_entities=final_entities[:30] if final_entities else None,
                        image_refs=final_images if final_images else None,
                        thinking=final_thinking,
                        agent_steps=collected_steps if collected_steps else None,
                    )
                    db.add(assistant_row)
                    await db.commit()
                except Exception as e:
                    logger.warning(f"Failed to persist assistant message: {e}")
                    await db.rollback()

    return StreamingResponse(
        sse_with_heartbeat(event_generator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
