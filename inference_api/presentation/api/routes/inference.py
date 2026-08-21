from fastapi import APIRouter, Depends
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dependency_injector.wiring import inject, Provide

from presentation.api.schemas import InferenceRequest, InferenceResponse
from application.use_cases.inference import ProcessImagesUseCase
from application.dtos import InferenceParametersDTO
from infrastructure.di_container import Container

logger = logging.getLogger(__name__)

router = APIRouter()
inference_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/", response_model=InferenceResponse)
@inject
async def predict(
    request: InferenceRequest,
    use_case: ProcessImagesUseCase = Depends(
        Provide[Container.process_images_use_case]
    ),
):
    tt = time.perf_counter()
    max_faces_param = request.parameters.max_faces
    max_faces = 0 if max_faces_param == "all" else int(max_faces_param)

    result = await asyncio.get_running_loop().run_in_executor(
        inference_executor,
        use_case.execute,
        request.inputs,
        InferenceParametersDTO(
            max_faces=max_faces,
            detection_conf=request.parameters.detection_conf,
            nms_threshold=request.parameters.nms_threshold
        ),
    )

    tt = time.perf_counter() - tt
    logger.info(f"Time taken: {tt}")
    return result
