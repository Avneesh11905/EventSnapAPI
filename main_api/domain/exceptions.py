class EventSnapError(Exception):
    """Base exception for EventSnap domain errors."""
    pass

class EventNotFoundError(EventSnapError):
    pass

class EncodingFailedError(EventSnapError):
    pass

class InvalidReferenceImagesError(EventSnapError):
    pass
