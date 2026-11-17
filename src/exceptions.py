"""
Domain exceptions for PulseForge AI.

All application-specific exceptions inherit from PulseForgeError.
These are caught at service/boundary layers to display actionable,
user-friendly messages without exposing internal stack traces.
"""

from __future__ import annotations


class PulseForgeError(Exception):
    """Base exception for all PulseForge AI errors."""

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message


# ---------------------------------------------------------------------------
# Audio / I/O Exceptions
# ---------------------------------------------------------------------------

class AudioValidationError(PulseForgeError):
    """Raised when an audio input fails safety or format validation."""


class FileTooLargeError(AudioValidationError):
    """Raised when uploaded file exceeds the configured byte size limit."""

    def __init__(self, file_size: int, max_size: int) -> None:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        msg = f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.2f} MB."
        user_msg = f"Uploaded file is too large ({size_mb:.1f} MB). Maximum allowed size is {max_mb:.0f} MB."
        super().__init__(msg, user_message=user_msg)
        self.file_size = file_size
        self.max_size = max_size


class InvalidFormatError(AudioValidationError):
    """Raised when uploaded file is not a valid or supported audio format."""

    def __init__(self, detected_format: str, supported_formats: tuple[str, ...]) -> None:
        msg = f"Unsupported audio format '{detected_format}'. Supported formats: {', '.join(supported_formats)}."
        user_msg = f"The uploaded file is not a valid WAV file. Please upload an uncompressed WAV audio file."
        super().__init__(msg, user_message=user_msg)
        self.detected_format = detected_format
        self.supported_formats = supported_formats


class AudioDurationError(AudioValidationError):
    """Raised when audio duration falls outside supported bounds."""


class AudioTooShortError(AudioDurationError):
    """Raised when audio duration is shorter than the minimum analysis window."""

    def __init__(self, duration: float, min_duration: float) -> None:
        msg = f"Audio duration ({duration:.2f}s) is shorter than minimum required ({min_duration:.2f}s)."
        user_msg = f"Recording is too short ({duration:.1f}s). Please provide a recording of at least {min_duration:.1f} seconds."
        super().__init__(msg, user_message=user_msg)
        self.duration = duration
        self.min_duration = min_duration


class AudioTooLongError(AudioDurationError):
    """Raised when audio duration exceeds maximum analysis limit."""

    def __init__(self, duration: float, max_duration: float) -> None:
        msg = f"Audio duration ({duration:.2f}s) exceeds maximum allowed ({max_duration:.2f}s)."
        user_msg = f"Recording is too long ({duration:.1f}s). Maximum supported duration is {max_duration:.0f} seconds."
        super().__init__(msg, user_message=user_msg)
        self.duration = duration
        self.max_duration = max_duration


class CorruptAudioError(AudioValidationError):
    """Raised when audio data cannot be decoded by soundfile/librosa."""

    def __init__(self, detail: str) -> None:
        msg = f"Audio file is corrupt or unreadable: {detail}"
        user_msg = "The audio file appears corrupted or unreadable. Please check the file and try again."
        super().__init__(msg, user_message=user_msg)
        self.detail = detail


# ---------------------------------------------------------------------------
# Model / Feature Exceptions
# ---------------------------------------------------------------------------

class ModelError(PulseForgeError):
    """Base exception for model artifact or inference failures."""


class ModelNotFoundError(ModelError):
    """Raised when requested model weights or configuration are missing."""

    def __init__(self, path_or_version: str) -> None:
        msg = f"Model artifact not found: {path_or_version}"
        user_msg = "Anomaly detection model is not loaded or could not be found."
        super().__init__(msg, user_message=user_msg)


class FeatureExtractionError(PulseForgeError):
    """Raised when feature extraction fails or produces non-finite values."""
