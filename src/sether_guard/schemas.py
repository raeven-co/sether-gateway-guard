"""OpenAI-compatible chat completion shapes: the subset the guard inspects.

Every model allows extra fields, so request parameters and response fields the
guard does not inspect pass through unchanged. Fields that can carry PII or
drive tool execution are typed, and anything the guard cannot inspect safely
is rejected rather than passed through.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _PassThrough(BaseModel):
    model_config = ConfigDict(extra="allow")


class FunctionCall(_PassThrough):
    name: str
    arguments: str


class ToolCall(_PassThrough):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


type ContentPart = dict[str, object]


class ResponseFormat(_PassThrough):
    type: str


_JSON_RESPONSE_FORMATS = frozenset({"json_object", "json_schema"})


class ChatMessage(_PassThrough):
    role: str
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionRequest(_PassThrough):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    response_format: ResponseFormat | None = None

    @property
    def wants_json_content(self) -> bool:
        """JSON mode: assistant content is a JSON document and must be restored as one."""
        return (
            self.response_format is not None and self.response_format.type in _JSON_RESPONSE_FORMATS
        )


class AssistantMessage(_PassThrough):
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    function_call: FunctionCall | None = None
    """Legacy single function call. Typed only so the guard can refuse it."""


class Choice(_PassThrough):
    index: int
    message: AssistantMessage
    finish_reason: str | None = None


class Usage(_PassThrough):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatCompletion(_PassThrough):
    id: str
    model: str
    choices: list[Choice]
    usage: Usage | None = None
