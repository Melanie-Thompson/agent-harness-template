from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from ai.context import build_context
from ai.guardrails import check_input, check_output
from ai.llm import get_llm
from ai.tools.base import get_tool
from config import Config
from db.conversations import add_message
from schemas.agent import AgentEvent
from schemas.conversation import ConversationMessage, Role

_pending_confirmations: dict[str, asyncio.Future[bool]] = {}


def _make_message_id() -> str:
    return f"{int(datetime.now(timezone.utc).timestamp() * 1000):013d}-{uuid4().hex[:12]}"


async def confirm_pending(pending_id: str, approved: bool) -> bool:
    future = _pending_confirmations.pop(pending_id, None)
    if future is None:
        return False
    future.set_result(approved)
    return True


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

    input_check = await check_input(prompt)
    if not input_check.safe:
        yield AgentEvent(type="blocked", data={"reason": input_check.reason, "stage": "input"})
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

        ai_message: AIMessage = await tool_llm.ainvoke(messages)
        messages.append(ai_message)

        if not ai_message.tool_calls:
            content = ai_message.content or ""
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)

            output_check = await check_output(content)
            if not output_check.safe:
                yield AgentEvent(type="blocked", data={"reason": output_check.reason, "stage": "output"})
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

            tool_instance = get_tool(tool_name)
            if tool_instance is None:
                error_msg = f"Tool '{tool_name}' not found"
                messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call_id, status="error"))
                yield AgentEvent(type="error", data={"tool": tool_name, "error": error_msg})
                continue

            yield AgentEvent(type="tool_call", data={"tool": tool_name, "args": tool_args})

            is_gated = tool_instance.metadata.get("gated", False)

            if is_gated:
                pending_id = uuid4().hex
                loop = asyncio.get_running_loop()
                future: asyncio.Future[bool] = loop.create_future()
                _pending_confirmations[pending_id] = future

                yield AgentEvent(
                    type="tool_pending",
                    data={"pending_id": pending_id, "tool": tool_name, "args": tool_args},
                )

                try:
                    approved = await asyncio.wait_for(future, timeout=300)
                except asyncio.TimeoutError:
                    _pending_confirmations.pop(pending_id, None)
                    approved = False

                if not approved:
                    deny_msg = "User denied this action"
                    messages.append(ToolMessage(content=deny_msg, tool_call_id=tool_call_id, status="error"))
                    yield AgentEvent(type="tool_denied", data={"pending_id": pending_id, "tool": tool_name})
                    continue

                yield AgentEvent(type="tool_confirmed", data={"pending_id": pending_id, "tool": tool_name})

            try:
                result = await tool_instance.ainvoke(tool_args)
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
            except Exception as exc:
                result = f"Tool error: {exc}"
                messages.append(ToolMessage(content=result, tool_call_id=tool_call_id, status="error"))
                yield AgentEvent(type="error", data={"tool": tool_name, "error": result})
                continue

            messages.append(ToolMessage(content=result, tool_call_id=tool_call_id))

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

    yield AgentEvent(type="error", data={"error": f"Agent reached maximum iterations ({max_iterations})"})
