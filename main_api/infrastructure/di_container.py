from dependency_injector import containers, providers
from application.use_cases.events import StartEventEncodingUseCase, CheckEncodingStatusUseCase, GetEncodedCountUseCase, DeleteEventTableUseCase
from application.use_cases.attendees import EncodeAttendeeUseCase, SortAttendeeUseCase, GenerateZipUseCase, CheckZipExistsUseCase
from application.use_cases.background_tasks import ProcessEventEncodingUseCase, CreateEventZipUseCase
from infrastructure.database.repository import PostgresEventRepository
from infrastructure.storage.minio_service import MinioStorageService
from infrastructure.inference.hf_inference_service import HFInferenceService
from infrastructure.queue.celery_service import CeleryTaskQueueService
from sqlalchemy.ext.asyncio import create_async_engine
from config import settings
from sqlalchemy.pool import NullPool

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "presentation.api.routers.events",
            "presentation.api.routers.attendees",
            "infrastructure.queue.celery_workers"
        ]
    )

    config = providers.Configuration()
    config.from_dict({
        "db_url": settings.DATABASE_URL,
        "minio_endpoint": settings.MINIO_ENDPOINT,
        "minio_bucket": settings.MINIO_BUCKET_NAME,
        "minio_access": settings.MINIO_ACCESS_KEY,
        "minio_secret": settings.MINIO_SECRET_KEY,
        "inference_url": settings.INFERENCE_API_URL,
        "inference_token": settings.INFERENCE_API_TOKEN,
    })

    db_engine = providers.Singleton(
        create_async_engine,
        config.db_url,
        poolclass=NullPool
    )

    event_repository = providers.Factory(
        PostgresEventRepository,
        engine=db_engine
    )

    storage_service = providers.Factory(
        MinioStorageService,
        endpoint_url=config.minio_endpoint,
        bucket_name=config.minio_bucket,
        access_key=config.minio_access,
        secret_key=config.minio_secret
    )

    inference_service = providers.Factory(
        HFInferenceService,
        api_url=config.inference_url,
        api_token=config.inference_token
    )

    queue_service = providers.Factory(
        CeleryTaskQueueService
    )

    start_event_encoding_use_case = providers.Factory(
        StartEventEncodingUseCase,
        queue_service=queue_service
    )
    
    check_encoding_status_use_case = providers.Factory(
        CheckEncodingStatusUseCase,
        queue_service=queue_service
    )
    
    get_encoded_count_use_case = providers.Factory(
        GetEncodedCountUseCase,
        repository=event_repository
    )
    
    delete_event_table_use_case = providers.Factory(
        DeleteEventTableUseCase,
        repository=event_repository
    )

    encode_attendee_use_case = providers.Factory(
        EncodeAttendeeUseCase,
        inference_service=inference_service
    )
    
    sort_attendee_use_case = providers.Factory(
        SortAttendeeUseCase,
        repository=event_repository
    )
    
    generate_zip_use_case = providers.Factory(
        GenerateZipUseCase,
        queue_service=queue_service
    )
    
    check_zip_exists_use_case = providers.Factory(
        CheckZipExistsUseCase,
        storage_service=storage_service
    )

    process_event_encoding_use_case = providers.Factory(
        ProcessEventEncodingUseCase,
        storage_service=storage_service,
        inference_service=inference_service,
        repository=event_repository
    )
    
    create_event_zip_use_case = providers.Factory(
        CreateEventZipUseCase,
        storage_service=storage_service
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
