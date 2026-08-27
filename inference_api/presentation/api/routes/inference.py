import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from application.dtos import InferenceParametersDTO
from application.use_cases.inference import ProcessImagesUseCase
from infrastructure.di_container import Container
from presentation.api.schemas import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=InferenceResponse)
@inject
async def predict(
    request: InferenceRequest,
    use_case: ProcessImagesUseCase = Depends(Provide[Container.process_images_use_case]),
    executor: ThreadPoolExecutor = Depends(Provide[Container.inference_executor]),
):
    logger.info(f"[HTTP] Received request for {len(request.inputs)} images")
    tt = time.perf_counter()
    max_faces_param = request.parameters.max_faces
    max_faces = 0 if max_faces_param == "all" else int(max_faces_param)

    result = await asyncio.get_running_loop().run_in_executor(
        executor,
        use_case.execute,
        request.inputs,
        InferenceParametersDTO(
            max_faces=max_faces,
            detection_conf=request.parameters.detection_conf,
            nms_threshold=request.parameters.nms_threshold,
        ),
    )

    tt = time.perf_counter() - tt
    logger.info(f"[HTTP] Time taken: {tt}")
    return result
