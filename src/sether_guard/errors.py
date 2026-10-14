"""Errors the guard maps to HTTP responses. Messages never contain request data."""


class GuardError(Exception):
    """Base class. ``code`` is stable and safe to log; ``message`` is safe to return."""

    status_code = 500
    code = "guard_error"
    message = "The guard could not process the request."

    def __init__(self, detail: str | None = None) -> None:
        super().__init__(detail or self.message)
        if detail is not None:
            self.message = detail


class StreamingNotSupportedError(GuardError):
    status_code = 400
    code = "streaming_not_supported"
    message = "This guard does not support stream=true yet. Send a non-streaming request."


class RunNotFoundError(GuardError):
    status_code = 404
    code = "run_not_found"
    message = "Unknown, expired or closed run."


class RunLimitError(GuardError):
    status_code = 503
    code = "run_limit_reached"
    message = "Too many open runs. Close finished runs before opening new ones."


class VaultCapacityError(GuardError):
    status_code = 413
    code = "vault_capacity_exceeded"
    message = "Too many sensitive values for one vault."


class UpstreamUnavailableError(GuardError):
    status_code = 502
    code = "upstream_unavailable"
    message = "The model provider could not be reached."


class UpstreamContractError(GuardError):
    status_code = 502
    code = "upstream_contract_violation"
    message = "The model provider returned a response the guard cannot safely inspect."


class UpstreamRejectedError(GuardError):
    """The provider answered with an HTTP error. Its body is passed through unrestored."""

    code = "upstream_rejected"
    message = "The model provider rejected the request."

    def __init__(self, status_code: int, body: bytes, content_type: str) -> None:
        super().__init__()
        self.status_code = status_code
        self.body = body
        self.content_type = content_type
