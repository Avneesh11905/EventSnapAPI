import asyncio
from celery_app import celery_app
from sqlalchemy import text, Column, String, Float, insert
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
import uuid
import re

from database import engine, SessionLocal, Base
from config import settings
from utils import list_minio_images, get_minio_session, download_minio_image_b64, call_inference_api

def sanitize_folder_path(path: str) -> str:
    """Convert 'events/summer_fest_2026' to 'events_summer_fest_2026'."""
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', path)
    return clean.strip('_')

def create_event_table_model(folder_path: str):
    """Dynamically creates a SQLAlchemy ORM Model for an Event."""
    table_name = f"event_{sanitize_folder_path(folder_path)}"
    
    # Class registry check to avoid redefining the same class
    if table_name in Base.metadata.tables:
        class EventModel(Base):
            __table__ = Base.metadata.tables[table_name]
        return EventModel
        
    class EventModel(Base):
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}
        
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        image_path = Column(String, index=True)
        # Assuming model outputs 512 dimensions, adjust if necessary
        embedding = Column(Vector(512))
        confidence = Column(Float)
        
    return EventModel

from botocore.config import Config

async def async_encode_event(self, folder_path: str, max_faces: int, det_conf: float, nms_thresh: float):
    """The core Async orchestration logic for the Celery Worker."""
    self.update_state(state='INITIALIZING', meta={'progress': 0, 'status': 'Listing MinIO files...'})
    
    # 1. Create DB Table
    EventModel = create_event_table_model(folder_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 2. Fetch all keys in folder
    all_keys = await list_minio_images(folder_path)
    
    if len(all_keys) == 0:
        return {"result": "No images found in folder.", "total": 0}
    
    # 2b. Query DB for already-encoded image paths and skip them
    already_encoded = set()
    table_name = f"event_{sanitize_folder_path(folder_path)}"
    async with SessionLocal() as db:
        result = await db.execute(
            text(f'SELECT DISTINCT image_path FROM "{table_name}"')
        )
        already_encoded = {row[0] for row in result.fetchall()}
    
    new_keys = [k for k in all_keys if k not in already_encoded]
    skipped = len(all_keys) - len(new_keys)
    total_images = len(new_keys)
    
    if total_images == 0:
        return {"result": "All images already encoded.", "total": len(all_keys), "skipped": skipped}
    
    self.update_state(state='PROCESSING', meta={
        'progress': 0, 'processed': 0, 'total': total_images,
        'skipped': skipped, 'status': f'Skipped {skipped} already-encoded images. Processing {total_images} new images...'
    })
        
    # 3. Pipeline process (Producer/Consumer Queue)
    batch_size = 64
    processed_count = 0
    download_queue = asyncio.Queue(maxsize=4) # Buffer up to 4 batches (256 images) in RAM
    
    s3_config = Config(max_pool_connections=64)
    # Re-use s3 connection pool per worker task
    async with get_minio_session().client('s3', endpoint_url=settings.MINIO_ENDPOINT, config=s3_config) as s3_client:
        
        async def minio_downloader():
            """Producer: Fetches images constantly and stuffs them into the RAM queue."""
            for i in range(0, total_images, batch_size):
                batch_keys = new_keys[i : i + batch_size]
                download_tasks = [download_minio_image_b64(s3_client, key) for key in batch_keys]
                b64_images = await asyncio.gather(*download_tasks)
                
                valid_pairs = [(key, b64) for key, b64 in zip(batch_keys, b64_images) if b64 is not None]
                if valid_pairs:
                    await download_queue.put(valid_pairs)
            
            # Sentinel to finish
            await download_queue.put(None)
            
        async def hf_inferencer():
            """Consumer: Drains the RAM queue and keeps the GPU pegged."""
            nonlocal processed_count
            while True:
                batch = await download_queue.get()
                if batch is None:
                    download_queue.task_done()
                    break
                    
                keys_to_process = [pair[0] for pair in batch]
                base64_strings = [pair[1] for pair in batch]
                
                try:
                    # Execute Inference
                    results = await call_inference_api(base64_strings, max_faces, det_conf, nms_thresh)
                    
                    # Prepare inserts
                    insert_data = []
                    for key, image_faces in zip(keys_to_process, results):
                        for face in image_faces:
                            emb = face.get("embedding")
                            conf = face.get("confidence")
                            if emb and conf:
                                insert_data.append({
                                    "id": uuid.uuid4(),
                                    "image_path": key,
                                    "embedding": emb,
                                    "confidence": conf
                                })
                                
                    # Bulk insert
                    if insert_data:
                        async with SessionLocal() as db:
                            await db.execute(insert(EventModel), insert_data)
                            await db.commit()
                            
                except Exception as e:
                    print(f"Failed to infer batch: {e}")
                    
                processed_count += len(keys_to_process)
                progress_pct = int((processed_count / total_images) * 100)
                
                self.update_state(
                    state='PROCESSING',
                    meta={
                        'progress': progress_pct,
                        'processed': processed_count,
                        'total': total_images,
                        'skipped': skipped,
                        'status': f'Processed {processed_count}/{total_images} new images (skipped {skipped} already encoded)'
                    }
                )
                download_queue.task_done()

        # Run Producer and Consumer concurrently
        await asyncio.gather(minio_downloader(), hf_inferencer())

@celery_app.task(bind=True, name="encode_event_task")
def encode_event_task(self, folder_path: str, max_faces: int = 0, det_conf: float = 0.5, nms_thresh: float = 0.4):
    """
    Celery entrypoint. Since Celery worker processes are synchronous, 
    we must boot up an asyncio event loop here to orchestrate the async I/O.
    """
    return asyncio.run(async_encode_event(self, folder_path, max_faces, det_conf, nms_thresh))
