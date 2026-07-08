from typing import List
from application.ports.inference import InferenceService
from application.ports.repository import EventRepository
from application.ports.queue import TaskQueueService
from application.ports.storage import StorageService
import base64
import io
from PIL import Image, ImageEnhance
import asyncio

class EncodeAttendeeUseCase:
    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service

    def _process_and_augment_images(self, b64_images: List[str]) -> List[str]:
        augmented_b64_images = []
        for b64_str in b64_images:
            image_data = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(image_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            augmented_b64_images.append(b64_str)
            
            enhancer = ImageEnhance.Sharpness(img)
            sharp_img = enhancer.enhance(2.0)
            buf1 = io.BytesIO()
            sharp_img.save(buf1, format="JPEG", quality=90)
            augmented_b64_images.append(base64.b64encode(buf1.getvalue()).decode('utf-8'))
            
            enhancer = ImageEnhance.Brightness(img)
            bright_img = enhancer.enhance(1.2)
            buf2 = io.BytesIO()
            bright_img.save(buf2, format="JPEG", quality=90)
            augmented_b64_images.append(base64.b64encode(buf2.getvalue()).decode('utf-8'))
            
        return augmented_b64_images

    async def execute(self, attendee_images_base64: List[str]) -> List[List[float]]:
        if len(attendee_images_base64) != 3:
            raise ValueError("Must provide exactly 3 attendee images (front, left, right).")
            
        augmented_b64_images = await asyncio.to_thread(self._process_and_augment_images, attendee_images_base64)
        
        results = await self.inference_service.get_face_encodings(augmented_b64_images)
            
        embeddings_list = []
        for image_faces in results:
            if len(image_faces) == 1:
                embeddings_list.append(image_faces[0]["embedding"])
                
        if not embeddings_list:
            raise ValueError("Could not detect clear faces in the provided and augmented reference images.")
            
        return embeddings_list

class SortAttendeeUseCase:
    def __init__(self, repository: EventRepository):
        self.repository = repository

    async def execute(self, minio_folder_path: str, attendee_encodings: List[List[float]]) -> dict:
        if len(attendee_encodings) == 0:
            raise ValueError("Must provide at least one attendee encoding.")
            
        has_table = await self.repository.check_table_exists(minio_folder_path)
        if not has_table:
             raise ValueError(f"No encoded data found for event {minio_folder_path}.")
             
        SIMILARITY_THRESHOLD = 0.55
        MIN_MATCHES = 2
        
        matched_paths = await self.repository.find_matches(
            minio_folder_path, attendee_encodings, SIMILARITY_THRESHOLD, MIN_MATCHES
        )

        if not matched_paths:
            debug_result = await self.repository.get_closest_matches_debug(minio_folder_path, attendee_encodings, 5)
            print("Closest 5 images (ignoring thresholds):", debug_result)
        
        return {
            "event": minio_folder_path,
            "matches_found": len(matched_paths),
            "photos": matched_paths
        }

class GenerateZipUseCase:
    def __init__(self, queue_service: TaskQueueService):
        self.queue_service = queue_service

    def execute(self, event_id: str, user_id: str, image_paths: list[dict]) -> str:
        return self.queue_service.enqueue_create_zip(event_id, user_id, image_paths)

class CheckZipExistsUseCase:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service

    async def execute(self, event_id: str, user_id: str) -> dict:
        zip_key = f"zips/{event_id}/{user_id}.zip"
        exists = await self.storage_service.check_zip_exists(zip_key)
        if exists:
            return {
                "exists": True,
                "zip_path": zip_key,
                "filename": f"{user_id}.zip"
            }
        return {"exists": False}
