class InferenceAPIError(Exception):
    """Base exception for Inference API errors."""


class InvalidInputFormatError(InferenceAPIError):
    pass


class ModelExecutionError(InferenceAPIError):
    pass
