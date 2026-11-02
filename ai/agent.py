from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import time

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from ai.context import build_context
from ai.evals.pipeline import run_eval_stage
from ai.llm import get_llm
from ai.pending import cleanup_pending, create_pending
from config import Config
from db.conversations import add_message
from schemas.agent import AgentEvent
from schemas.ai import EvalContext, EvalStage, EvalVerdict
from logging_config import get_logger
from logging_decorator import log_call
from schemas.conversation import ConversationMessage, Role

_log = get_logger("ai.agent")

_BLOCK_SUMMARY_PROMPT = (
    "A user's message was blocked by a safety guardrail. "
    "The matched safety category was: \"{category}\"\n"
    "The user's message was: \"{user_input}\"\n\n"
    "Write a brief, friendly 1-2 sentence explanation to the user about why their "
    "message was blocked. Do not repeat the blocked content. Do not lecture. "
    "Be concise and helpful."
)


async def _summarise_block_reason(
    category: str,
    user_input: str,
    model_override: str | None = None,
) -> str:
    try:
        llm = get_llm(model_override=model_override)
        prompt = _BLOCK_SUMMARY_PROMPT.format(
            category=category,
            user_input=user_input[:200],
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content or ""
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        return content.strip()
    except Exception as exc:
        _log.error("agent.block_summary_error", error=str(exc))
        return category


def _make_message_id() -> str:
    return f"{int(datetime.now(timezone.utc).timestamp() * 1000):013d}-{uuid4().hex[:12]}"


@log_call
async def invoke_agent(
    prompt: str,
    *,
    tools: list[BaseTool],
    user_id: str,
    session_id: str | None = None,
    conversation_id: str | None = None,
    model: str | None = None,
    guardrail_model: str | None = None,
    max_iterations: int | None = None,
) -> AsyncGenerator[AgentEvent, None]:
    max_iterations = max_iterations or Config.AGENT_MAX_ITERATIONS
    tool_map = {t.name: t for t in tools}

    input_eval = await run_eval_stage(
        EvalStage.INPUT, EvalContext(user_id=user_id, stage=EvalStage.INPUT, prompt=prompt),
    )
    _log.info("agent.input_eval", verdict=input_eval.verdict.value, reason=input_eval.reason)
    if input_eval.verdict != EvalVerdict.PASS:
        summary = await _summarise_block_reason(
            input_eval.reason or "safety policy",
            prompt,
            model_override=model,
        )
        yield AgentEvent(type="blocked", data={
            "reason": summary,
            "category": input_eval.reason,
            "stage": "input",
        })
        return

    llm = get_llm(model_override=model)
    tool_llm = llm.bind_tools(tools) if tools else llm

    messages: list[Any] = []

    if Config.AGENT_SYSTEM_PROMPT:
        messages.append(SystemMessage(content=Config.AGENT_SYSTEM_PROMPT))

    if conversation_id:
        history = await build_context(conversation_id)
        role_map = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
        for m in history:
            cls = role_map.get(m["role"], HumanMessage)
            messages.append(cls(content=m["content"]))

    messages.append(HumanMessage(content=prompt))

    if conversation_id and session_id:
        await add_message(ConversationMessage(
            conversation_id=conversation_id,
            message_id=_make_message_id(),
            session_id=session_id,
            user_id=user_id,
            role=Role.USER,
            timestamp=datetime.now(timezone.utc),
            content=prompt,
        ))

    for iteration in range(max_iterations):
        yield AgentEvent(type="thinking", data={"iteration": iteration + 1})

        t_llm = time.perf_counter()
        ai_message: AIMessage | None = None
        try:
            async for chunk in tool_llm.astream(messages):
                if ai_message is None:
                    ai_message = chunk
                else:
                    ai_message = ai_message + chunk
                if chunk.content and not getattr(chunk, "tool_call_chunks", None):
                    text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if text:
                        yield AgentEvent(type="token", data={"token": text})
        except Exception as exc:
            _log.error("agent.llm_stream_error", error=str(exc))
            yield AgentEvent(type="error", data={"error": f"LLM error: {exc}"})
            return
        if ai_message is None:
            ai_message = AIMessage(content="")
        _log.info("agent.llm_call", iteration=iteration + 1, duration_ms=round((time.perf_counter() - t_llm) * 1000, 2))
        messages.append(ai_message)

        if not ai_message.tool_calls:
            content = ai_message.content or ""
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)

            output_eval = await run_eval_stage(
                EvalStage.OUTPUT, EvalContext(user_id=user_id, stage=EvalStage.OUTPUT, output=content),
            )
            _log.info("agent.output_eval", verdict=output_eval.verdict.value)
            if output_eval.verdict != EvalVerdict.PASS:
                summary = await _summarise_block_reason(
                    output_eval.reason or "safety policy",
                    content[:200],
                    model_override=model,
                )
                yield AgentEvent(type="blocked", data={
                    "reason": summary,
                    "category": output_eval.reason,
                    "stage": "output",
                })
                return

            if conversation_id and session_id:
                await add_message(ConversationMessage(
                    conversation_id=conversation_id,
                    message_id=_make_message_id(),
                    session_id=session_id,
                    user_id=user_id,
                    role=Role.ASSISTANT,
                    timestamp=datetime.now(timezone.utc),
                    content=content,
                ))

            yield AgentEvent(type="response", data={"content": content})
            return

        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call.get("id", uuid4().hex)

            tool_instance = tool_map.get(tool_name)
            _log.info("agent.tool_lookup", tool=tool_name, found=tool_instance is not None)
            if tool_instance is None:
                error_msg = f"Tool '{tool_name}' not found"
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, status="error"))
                yield AgentEvent(type="error", data={"tool": tool_name, "error": error_msg})
                continue

            yield AgentEvent(type="tool_call", data={"tool": tool_name, "args": tool_args})

            is_gated = tool_instance.metadata.get("gated", False)

            if is_gated:
                pending_id = uuid4().hex
                future = await create_pending(
                    pending_id,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    user_id=user_id,
                )

                _log.info("agent.gated_pending", pending_id=pending_id, tool=tool_name)
                yield AgentEvent(
                    type="tool_pending",
                    data={"pending_id": pending_id, "tool": tool_name, "args": tool_args},
                )

                try:
                    approved = await asyncio.wait_for(future, timeout=Config.AGENT_PENDING_TTL)
                except asyncio.TimeoutError:
                    await cleanup_pending(pending_id)
                    approved = False

                _log.info("agent.gated_resolved", pending_id=pending_id, approved=approved)
                if not approved:
                    deny_msg = "User denied this action"
                    messages.append(ToolMessage(content=deny_msg, tool_call_id=tool_call_id, status="error"))
                    yield AgentEvent(type="tool_denied", data={"pending_id": pending_id, "tool": tool_name})
                    continue

                yield AgentEvent(type="tool_confirmed", data={"pending_id": pending_id, "tool": tool_name})

            t_tool = time.perf_counter()
            try:
                result = await tool_instance.ainvoke(tool_args)
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
            except Exception as exc:
                _log.error("agent.tool_error", tool=tool_name, error=str(exc), duration_ms=round((time.perf_counter() - t_tool) * 1000, 2))
                result = f"Tool error: {exc}"
                messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, status="error"))
                yield AgentEvent(type="error", data={"tool": tool_name, "error": result})
                continue

            _log.info("agent.tool_result", tool=tool_name, result_len=len(result), duration_ms=round((time.perf_counter() - t_tool) * 1000, 2))
            messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

            tool_eval = await run_eval_stage(
                EvalStage.TOOL_RESULT,
                EvalContext(
                    user_id=user_id,
                    stage=EvalStage.TOOL_RESULT,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=result,
                ),
            )
            _log.info("agent.tool_eval", tool=tool_name, verdict=tool_eval.verdict.value)
            if tool_eval.verdict == EvalVerdict.FAIL:
                messages[-1] = ToolMessage(
                    content=f"Tool result vetoed: {tool_eval.reason}",
                    tool_call_id=tool_call_id,
                    status="error",
                )
                yield AgentEvent(type="blocked", data={
                    "reason": tool_eval.reason, "stage": "tool_result", "tool": tool_name,
                })
                continue
            if tool_eval.verdict == EvalVerdict.RETRY:
                messages.pop()
                corrective = f"Previous result was rejected. {tool_eval.corrective_context or ''}"
                messages.append(ToolMessage(content=corrective, tool_call_id=tool_call_id, status="error"))
                yield AgentEvent(type="eval_retry", data={
                    "reason": tool_eval.reason, "stage": "tool_result", "tool": tool_name,
                })
                continue

            if conversation_id and session_id:
                await add_message(ConversationMessage(
                    conversation_id=conversation_id,
                    message_id=_make_message_id(),
                    session_id=session_id,
                    user_id=user_id,
                    role=Role.TOOL,
                    timestamp=datetime.now(timezone.utc),
                    content=result,
                    metadata={"tool": tool_name, "args": tool_args},
                ))

            yield AgentEvent(type="tool_result", data={"tool": tool_name, "result": result})

    _log.warning("agent.max_iterations", max_iterations=max_iterations)
    yield AgentEvent(type="error", data={"error": f"Agent reached maximum iterations ({max_iterations})"})
