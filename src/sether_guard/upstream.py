"""Forward a tokenized request to the OpenAI-compatible upstream."""

import httpx
from pydantic import ValidationError

from .errors import UpstreamContractError, UpstreamRejectedError, UpstreamUnavailableError
from .schemas import ChatCompletion, ChatCompletionRequest


class UpstreamClient:
    """Only the caller's ``Authorization`` header is forwarded; it is never stored or logged."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def create_chat_completion(
        self, request: ChatCompletionRequest, authorization: str | None
    ) -> ChatCompletion:
        response = await self._post(
            request.model_dump(mode="json", exclude_unset=True), authorization
        )
        if response.is_error:
            content_type = response.headers.get("content-type", "application/json")
            raise UpstreamRejectedError(response.status_code, response.content, content_type)
        try:
            return ChatCompletion.model_validate_json(response.content)
        except ValidationError as error:
            raise UpstreamContractError from error

    async def _post(self, payload: dict[str, object], authorization: str | None) -> httpx.Response:
        headers = {"authorization": authorization} if authorization else {}
        try:
            return await self._http.post("chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as error:
            raise UpstreamUnavailableError from error
