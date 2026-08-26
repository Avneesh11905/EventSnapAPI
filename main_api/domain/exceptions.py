class EventSnapError(Exception):
    """Base exception for EventSnap domain errors."""

    pass


class EventNotFoundError(EventSnapError):
    pass


class EncodingFailedError(EventSnapError):
    pass


class InvalidReferenceImagesError(EventSnapError):
    pass


class FaceValidationError(EventSnapError):
    def __init__(self, message: str, details: list | None = None):
        super().__init__(message)
        self.details = details or []


class ZipGenerationError(EventSnapError):
    pass


class StorageDownloadError(EventSnapError):
    pass


class InferenceError(EventSnapError):
    pass


class NoMatchesFoundError(EventSnapError):
    pass


class TaskAlreadyInProgressError(EventSnapError):
    pass


class TaskCanceledError(EventSnapError):
    pass
