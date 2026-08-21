from dependency_injector import containers, providers
from application.use_cases.events import (
    StartEventEncodingUseCase,
    CheckEncodingStatusUseCase,
    GetEncodedCountUseCase,
    DeleteEventDataUseCase,
)
from application.use_cases.attendees import (
    EncodeAttendeeUseCase,
    SortAttendeeUseCase,
    GenerateZipUseCase,
    CheckZipExistsUseCase,
)
from application.use_cases.background_tasks import (
    ProcessEventEncodingUseCase,
    CreateEventZipUseCase,
    EncodeImageBatchUseCase,
)
from infrastructure.database.uow import AsyncSqlAlchemyUnitOfWork
from infrastructure.storage.s3_service import S3StorageService
from infrastructure.inference.onnx_inference_service import OnnxInferenceService
from infrastructure.queue.celery_service import CeleryTaskQueueService
from infrastructure.image_augmenter import OpenCVImageAugmenter
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from config import settings
from sqlalchemy.pool import NullPool


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "presentation.api.routers.events",
            "presentation.api.routers.attendees",
            "infrastructure.queue.celery_workers",
        ]
    )

    config = providers.Configuration()
    config.from_dict(
        {
            "db_url": settings.DATABASE_URL,
            "storage_endpoint": settings.STORAGE_ENDPOINT,
            "storage_bucket": settings.STORAGE_BUCKET_NAME,
            "storage_access": settings.STORAGE_ACCESS_KEY,
            "storage_secret": settings.STORAGE_SECRET_KEY,
            "inference_url": settings.INFERENCE_API_URL,
        }
    )

    db_engine = providers.Singleton(
        create_async_engine, config.db_url, poolclass=NullPool
    )

    session_factory = providers.Singleton(
        async_sessionmaker, bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )

    uow = providers.Factory(AsyncSqlAlchemyUnitOfWork, session_factory=session_factory)

    storage_service = providers.Factory(
        S3StorageService,
        endpoint_url=config.storage_endpoint,
        bucket_name=config.storage_bucket,
        access_key=config.storage_access,
        secret_key=config.storage_secret,
    )

    inference_service = providers.Factory(
        OnnxInferenceService,
        api_url=config.inference_url,
    )

    queue_service = providers.Factory(CeleryTaskQueueService)

    image_augmenter = providers.Factory(OpenCVImageAugmenter)

    start_event_encoding_use_case = providers.Factory(
        StartEventEncodingUseCase, queue_service=queue_service
    )

    check_encoding_status_use_case = providers.Factory(
        CheckEncodingStatusUseCase, queue_service=queue_service
    )

    get_encoded_count_use_case = providers.Factory(GetEncodedCountUseCase, uow=uow)

    delete_event_data_use_case = providers.Factory(
        DeleteEventDataUseCase, queue_service=queue_service
    )

    encode_attendee_use_case = providers.Factory(
        EncodeAttendeeUseCase,
        inference_service=inference_service,
        augmenter=image_augmenter,
    )

    sort_attendee_use_case = providers.Factory(SortAttendeeUseCase, uow=uow)

    generate_zip_use_case = providers.Factory(
        GenerateZipUseCase, queue_service=queue_service
    )

    check_zip_exists_use_case = providers.Factory(
        CheckZipExistsUseCase, storage_service=storage_service
    )

    process_event_encoding_use_case = providers.Factory(
        ProcessEventEncodingUseCase,
        storage_service=storage_service,
        uow=uow,
        queue_service=queue_service,
    )

    encode_image_batch_use_case = providers.Factory(
        EncodeImageBatchUseCase,
        storage_service=storage_service,
        inference_service=inference_service,
        uow=uow,
    )

    create_event_zip_use_case = providers.Factory(
        CreateEventZipUseCase, storage_service=storage_service
    )


_container_instance = None


def get_container() -> Container:
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
        # Initial wiring is handled by DeclarativeContainer automatically if modules are passed
        # But we can force it here just in case:
        _container_instance.wire()
    return _container_instance
