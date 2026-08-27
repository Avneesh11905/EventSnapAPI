from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from application.use_cases.attendees import (
    CheckZipExistsUseCase,
    EncodeAttendeeUseCase,
    GenerateZipUseCase,
    SortAttendeeUseCase,
)
from application.use_cases.background_tasks import (
    CreateEventZipUseCase,
    DeleteImageBatchUseCase,
    EncodeImageBatchUseCase,
    ProcessEventEncodingUseCase,
)
from application.use_cases.events import (
    CheckEncodingStatusUseCase,
    DeleteEventDataUseCase,
    GetEncodedCountUseCase,
    StartEventEncodingUseCase,
)
from config.database import db_settings
from config.inference import inference_settings
from config.queue import queue_settings
from config.storage import storage_settings
from infrastructure.cache.valkey_service import ValkeyCacheService
from infrastructure.database.uow import AsyncSqlAlchemyUnitOfWork
from infrastructure.image_augmenter import OpenCVImageAugmenter
from infrastructure.inference.onnx_inference_service import OnnxInferenceService
from infrastructure.queue.celery_service import CeleryTaskQueueService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "presentation.api.routers.events",
            "presentation.api.routers.attendees",
            "presentation.api.routers.images",
            "infrastructure.queue.celery_workers",
        ]
    )

    config = providers.Configuration()
    config.from_dict(
        {
            "db_url": db_settings.DATABASE_URL,
            "storage_endpoint": storage_settings.STORAGE_ENDPOINT,
            "storage_bucket": storage_settings.STORAGE_BUCKET_NAME,
            "storage_access": storage_settings.STORAGE_ACCESS_KEY,
            "storage_secret": storage_settings.STORAGE_SECRET_KEY,
            "inference_url": inference_settings.INFERENCE_API_URL,
        }
    )

    db_engine = providers.Singleton(create_async_engine, config.db_url, poolclass=NullPool)

    session_factory = providers.Singleton(
        async_sessionmaker, bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )

    uow = providers.Factory(AsyncSqlAlchemyUnitOfWork, session_factory=session_factory)

    cache_service = providers.Factory(ValkeyCacheService, valkey_url=queue_settings.VALKEY_URL)

    queue_service = providers.Factory(CeleryTaskQueueService, cache_service=cache_service)

    image_augmenter = providers.Factory(OpenCVImageAugmenter)

    if inference_settings.INFERENCE_API_GRPC_URL:
        import base64

        from infrastructure.inference.grpc_inference_service import GrpcInferenceService
        from infrastructure.storage.s3_service import S3StorageServiceBytes

        storage_service = providers.Factory(
            S3StorageServiceBytes,
            endpoint_url=config.storage_endpoint,
            bucket_name=config.storage_bucket,
            access_key=config.storage_access,
            secret_key=config.storage_secret,
        )
        inference_service = providers.Factory(
            GrpcInferenceService,
            api_url=inference_settings.INFERENCE_API_GRPC_URL,
        )
        encode_attendee_use_case = providers.Factory(
            EncodeAttendeeUseCase[bytes],
            inference_service=inference_service,
            augmenter=image_augmenter,
            decode_fn=base64.b64decode,
        )
        encode_image_batch_use_case = providers.Factory(
            EncodeImageBatchUseCase[bytes],
            storage_service=storage_service,
            inference_service=inference_service,
            uow=uow,
            cache_service=cache_service,
        )
    else:
        from infrastructure.storage.s3_service import S3StorageServiceB64

        storage_service = providers.Factory(
            S3StorageServiceB64,  # type: ignore[arg-type]
            endpoint_url=config.storage_endpoint,
            bucket_name=config.storage_bucket,
            access_key=config.storage_access,
            secret_key=config.storage_secret,
        )
        inference_service = providers.Factory(
            OnnxInferenceService,  # type: ignore[arg-type]
            api_url=config.inference_url,
        )
        encode_attendee_use_case = providers.Factory(
            EncodeAttendeeUseCase[str],  # type: ignore[arg-type]
            inference_service=inference_service,
            augmenter=image_augmenter,
            decode_fn=lambda x: x,
        )
        encode_image_batch_use_case = providers.Factory(
            EncodeImageBatchUseCase[str],  # type: ignore[arg-type]
            storage_service=storage_service,
            inference_service=inference_service,
            uow=uow,
            cache_service=cache_service,
        )

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

    sort_attendee_use_case = providers.Factory(SortAttendeeUseCase, uow=uow)

    generate_zip_use_case = providers.Factory(GenerateZipUseCase, queue_service=queue_service)

    check_zip_exists_use_case = providers.Factory(
        CheckZipExistsUseCase, storage_service=storage_service
    )

    process_event_encoding_use_case = providers.Factory(
        ProcessEventEncodingUseCase,
        storage_service=storage_service,
        uow=uow,
        queue_service=queue_service,
        cache_service=cache_service,
    )

    create_event_zip_use_case = providers.Factory(
        CreateEventZipUseCase, storage_service=storage_service
    )

    delete_image_batch_use_case = providers.Factory(
        DeleteImageBatchUseCase,
        storage_service=storage_service,
        uow=uow,
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
