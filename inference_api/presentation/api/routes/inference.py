from fastapi import APIRouter, Depends
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dependency_injector.wiring import inject, Provide

from presentation.api.schemas import InferenceRequest
from application.use_cases.inference import ProcessImagesUseCase
from infrastructure.di_container import Container

logger = logging.getLogger(__name__)

router = APIRouter()
inference_executor = ThreadPoolExecutor(max_workers=1)


@router.post("/")
@inject
async def predict(
    request: InferenceRequest,
    use_case: ProcessImagesUseCase = Depends(
        Provide[Container.process_images_use_case]
    ),
):
    tt = time.perf_counter()

    result = await asyncio.get_running_loop().run_in_executor(
        inference_executor,
        use_case.execute,
        request.inputs,
        request.parameters.model_dump(),
    )

    tt = time.perf_counter() - tt
    logger.info(f"Time taken: {tt}")
    return result
