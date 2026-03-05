class TraceReaderError(Exception):
    """Base exception for offline trace reading."""


class TraceFormatError(TraceReaderError):
    """Raised when a trace file exists but does not match the expected format."""


class TraceNotFoundError(TraceReaderError):
    """Raised when a requested trace file or resource cannot be found."""


class UnsupportedAttentionTypeError(TraceReaderError):
    """Raised when an attention type is not supported by the reader."""