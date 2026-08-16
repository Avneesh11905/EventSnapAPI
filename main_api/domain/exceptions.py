class EventSnapError(Exception):
    """Base exception for EventSnap domain errors."""

    pass


class EventNotFoundError(EventSnapError):
    pass


class EncodingFailedError(EventSnapError):
    pass


class InvalidReferenceImagesError(EventSnapError):
    pass


class ZipGenerationError(EventSnapError):
    pass


class StorageDownloadError(EventSnapError):
    pass


class InferenceError(EventSnapError):
    pass


class NoMatchesFoundError(EventSnapError):
    pass
