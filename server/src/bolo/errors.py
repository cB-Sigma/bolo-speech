"""Exception types shared across the server.

Every error carries an HTTP status, a short machine-readable ``code`` and an
OpenAI-style ``type`` so each protocol front end can render it its own way.
"""

from __future__ import annotations


class BoloError(Exception):
    status = 500
    code = "internal_error"
    type = "server_error"

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


class InvalidRequest(BoloError):
    status = 400
    code = "invalid_request"
    type = "invalid_request_error"


class AuthError(BoloError):
    status = 401
    code = "invalid_api_key"
    type = "invalid_request_error"


class ModelNotFound(BoloError):
    status = 404
    code = "model_not_found"
    type = "invalid_request_error"


class ModelNotInstalled(BoloError):
    status = 404
    code = "model_not_installed"
    type = "invalid_request_error"


class ModelNotPublished(BoloError):
    """A catalog entry whose files have not been published yet (``status: pending``)."""

    status = 404
    code = "model_not_published"
    type = "invalid_request_error"


class GatedModel(BoloError):
    status = 401
    code = "model_gated"
    type = "invalid_request_error"


class EngineUnavailable(BoloError):
    status = 503
    code = "engine_unavailable"
    type = "server_error"


class ComponentMissing(BoloError):
    """An optional third-party component (espeak-ng) that the request needs is not installed."""

    status = 424
    code = "component_missing"
    type = "invalid_request_error"


class LLMNotConfigured(BoloError):
    status = 503
    code = "llm_not_configured"
    type = "server_error"


class LLMError(BoloError):
    status = 502
    code = "llm_error"
    type = "server_error"


class SynthesisFailed(BoloError):
    status = 500
    code = "synthesis_failed"
    type = "server_error"


class Overloaded(BoloError):
    """Raised when a bounded queue is full: the client should back off."""

    status = 429
    code = "overloaded"
    type = "rate_limit_error"

    def __init__(self, message: str, *, retry_after_s: float = 1.0) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class AudioDecodeError(InvalidRequest):
    code = "audio_decode_error"


class UnsupportedFormat(InvalidRequest):
    code = "unsupported_format"


class UnsupportedCharacters(InvalidRequest):
    """The text contains letters or digits that a character-input voice has no symbol for."""

    code = "unsupported_characters"


class DownloadError(BoloError):
    status = 502
    code = "download_failed"
    type = "server_error"
