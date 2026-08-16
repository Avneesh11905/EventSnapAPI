class InferenceAPIError(Exception):
    """Base exception for Inference API errors."""

    pass


class InvalidInputFormatError(InferenceAPIError):
    pass


class ModelExecutionError(InferenceAPIError):
    pass
