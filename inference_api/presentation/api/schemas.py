from pydantic import BaseModel, Field


class InferenceParameters(BaseModel):
    max_faces: int | str = 0
    detection_conf: float = 0.5
    nms_threshold: float = 0.4


class InferenceRequest(BaseModel):
    inputs: list[str] | str
    parameters: InferenceParameters = Field(default_factory=InferenceParameters)
