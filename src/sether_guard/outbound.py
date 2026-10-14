"""Outbound: tokenize every message before it leaves for the provider.

Covers string content and text parts for every role (system, developer, user,
assistant, tool) and the JSON arguments of assistant tool calls in history.
History matters: after turn 1 the caller's transcript holds *restored* tool
arguments, so turn 2 would leak them if only ``content`` were tokenized.

Each message is redacted with a function bound to its role, so the vault can
record where every token came from (see ``token_fields.sources`` in the policy).
"""

from collections.abc import Callable

from .jsonwalk import dumps_json, map_all_strings, parse_json_object
from .schemas import ChatCompletionRequest, ChatMessage, ContentPart, ToolCall

type Redact = Callable[[str], str]
type RedactFor = Callable[[str], Redact]
"""Given a message role, return a redact function that attributes tokens to it."""


def tokenize_request(
    request: ChatCompletionRequest, redact_for: RedactFor
) -> ChatCompletionRequest:
    messages = [tokenize_message(m, redact_for(m.role)) for m in request.messages]
    return request.model_copy(update={"messages": messages})


def tokenize_message(message: ChatMessage, redact: Redact) -> ChatMessage:
    update: dict[str, object] = {}
    if message.content is not None:
        update["content"] = _tokenize_content(message.content, redact)
    if message.tool_calls is not None:
        update["tool_calls"] = [_tokenize_tool_call(call, redact) for call in message.tool_calls]
    return message.model_copy(update=update)


def _tokenize_content(content: str | list[ContentPart], redact: Redact) -> str | list[ContentPart]:
    if isinstance(content, str):
        return redact(content)
    return [_tokenize_part(part, redact) for part in content]


def _tokenize_part(part: ContentPart, redact: Redact) -> ContentPart:
    text = part.get("text")
    if part.get("type") == "text" and isinstance(text, str):
        return {**part, "text": redact(text)}
    return part


def _tokenize_tool_call(call: ToolCall, redact: Redact) -> ToolCall:
    raw = call.function.arguments
    parsed = parse_json_object(raw)
    # Redact decoded strings, then re-serialize: redacting raw JSON text could
    # split an escape sequence. Unparseable arguments are redacted as plain text.
    arguments = redact(raw) if parsed is None else dumps_json(map_all_strings(parsed, redact))
    function = call.function.model_copy(update={"arguments": arguments})
    return call.model_copy(update={"function": function})
