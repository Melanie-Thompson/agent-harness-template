from config import Config
from db.conversations import get_messages
from logging_decorator import log_call
from schemas.conversation import ConversationMessage


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


@log_call
async def build_context(
    conversation_id: str,
    *,
    max_messages: int = 0,
    max_tokens: int = 0,
) -> list[dict]:
    max_messages = max_messages or Config.CONTEXT_MAX_MESSAGES
    max_tokens = max_tokens or Config.CONTEXT_MAX_TOKENS

    messages = await get_messages(conversation_id, limit=max_messages)

    trimmed: list[dict] = []
    token_budget = max_tokens

    for msg in reversed(messages):
        cost = _estimate_tokens(msg.content)
        if cost > token_budget:
            break
        trimmed.append({"role": msg.role.value, "content": msg.content})
        token_budget -= cost

    trimmed.reverse()
    return trimmed
