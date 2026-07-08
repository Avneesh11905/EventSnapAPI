from application.ports.storage import StorageService
from application.ports.inference import InferenceService
from application.ports.repository import EventRepository
import asyncio
from typing import Callable, Any
import uuid

class ProcessEventEncodingUseCase:
    def __init__(self, storage_service: StorageService, inference_service: InferenceService, repository: EventRepository):
        self.storage_service = storage_service
        self.inference_service = inference_service
        self.repository = repository

    async def execute(self, folder_path: str, max_faces: int, det_conf: float, nms_thresh: float, update_state_cb: Callable[[str, dict], Any]):
        update_state_cb('INITIALIZING', {'progress': 0, 'status': 'Listing MinIO files...'})
        
        await self.repository.create_event_table(folder_path)
        
        all_keys = await self.storage_service.list_images(folder_path)
        
        if len(all_keys) == 0:
            return {"result": "No images found in folder.", "total": 0}
            
        already_encoded = await self.repository.get_already_encoded_images(folder_path)
        
        new_keys = [k for k in all_keys if k not in already_encoded]
        skipped = len(all_keys) - len(new_keys)
        total_images = len(new_keys)
        
        if total_images == 0:
            return {"result": "All images already encoded.", "total": len(all_keys), "skipped": skipped}
            
        update_state_cb('PROCESSING', {
            'progress': 0, 'processed': 0, 'total': total_images,
            'skipped': skipped, 'status_msg': f'Skipped {skipped} already-encoded images. Processing {total_images} new images...'
        })

        batch_size = 64
        processed_count = 0
        download_queue = asyncio.Queue(maxsize=4) 
        
        async def minio_downloader():
            for i in range(0, total_images, batch_size):
                batch_keys = new_keys[i : i + batch_size]
                download_tasks = [self.storage_service.download_image_b64(key) for key in batch_keys]
                b64_images = await asyncio.gather(*download_tasks, return_exceptions=True)
                
                valid_pairs = []
                for key, b64 in zip(batch_keys, b64_images):
                    if isinstance(b64, Exception):
                        print(f"Failed to download {key} after retries: {b64}")
                    elif b64 is not None:
                        valid_pairs.append((key, b64))

                if valid_pairs:
                    await download_queue.put(valid_pairs)
            
            await download_queue.put(None)
            
        async def hf_inferencer():
            nonlocal processed_count
            while True:
                batch = await download_queue.get()
                if batch is None:
                    download_queue.task_done()
                    break
                    
                keys_to_process = [pair[0] for pair in batch]
                base64_strings = [pair[1] for pair in batch]
                
                try:
                    results = await self.inference_service.get_face_encodings(base64_strings, max_faces, det_conf, nms_thresh)
                    
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
                                
                    if insert_data:
                        await self.repository.save_encodings(folder_path, insert_data)
                            
                except Exception as e:
                    print(f"Failed to infer batch: {e}")
                    
                processed_count += len(keys_to_process)
                progress_pct = int((processed_count / total_images) * 100)
                
                update_state_cb('PROCESSING', {
                    'progress': progress_pct,
                    'processed': processed_count,
                    'total': total_images,
                    'skipped': skipped,
                    'status_msg': f'Processed {processed_count}/{total_images} new images (skipped {skipped} already encoded)'
                })
                download_queue.task_done()

        await asyncio.gather(minio_downloader(), hf_inferencer())
        return {"result": "Encoding finished successfully.", "total": len(all_keys), "skipped": skipped}


class CreateEventZipUseCase:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service

    async def execute(self, event_id: str, user_id: str, image_paths: list[dict], update_state_cb: Callable[[str, dict], Any]):
        total = len(image_paths)
        if total == 0:
            return {"error": "No images to zip"}

        update_state_cb('INITIALIZING', {'progress': 0, 'status_msg': f'Starting ZIP for {total} images...'})

        zip_filename = f"{user_id}.zip"
        storage_path = f"zips/{event_id}/{zip_filename}"
        
        def progress_callback(current, state_name, status_msg):
            progress_pct = int((current / total) * 90)
            update_state_cb(state_name, {
                'progress': progress_pct,
                'status_msg': status_msg
            })

        try:
            await self.storage_service.create_zip_from_images(storage_path, image_paths, progress_callback)
            
            return {
                "status": "COMPLETED",
                "progress": 100,
                "zip_path": storage_path,
                "filename": zip_filename
            }
        except Exception as e:
            update_state_cb('FAILED', {'error': str(e)})
            raise e
