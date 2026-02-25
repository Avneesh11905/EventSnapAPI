from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import base64
import io
from PIL import Image, ImageEnhance

from database import get_db, engine
from utils import call_inference_api
from tasks import create_event_table_model # Re-use the schema gen

router = APIRouter()

class EncodeAttendeeRequest(BaseModel):
    attendee_images_base64: list[str]

class SortAttendeeRequest(BaseModel):
    minio_folder_path: str
    attendee_encodings: list[list[float]]

def process_and_augment_images(b64_images: list[str]) -> list[str]:
    """Synchronous CPU-bound image a    ugmentation function meant to run in a thread pool."""
    augmented_b64_images = []
    
    for b64_str in b64_images:
        # 1. Decode original image
        image_data = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Keep original
        augmented_b64_images.append(b64_str)
        
        # Augmentation 1: Sharpness Enhance (improves feature extraction on soft faces)
        enhancer = ImageEnhance.Sharpness(img)
        sharp_img = enhancer.enhance(2.0)
        buf1 = io.BytesIO()
        sharp_img.save(buf1, format="JPEG", quality=90)
        augmented_b64_images.append(base64.b64encode(buf1.getvalue()).decode('utf-8'))
        
        # Augmentation 2: Brightness Adjustment (e.g., 1.2x brighter)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(1.2)
        buf2 = io.BytesIO()
        bright_img.save(buf2, format="JPEG", quality=90)
        augmented_b64_images.append(base64.b64encode(buf2.getvalue()).decode('utf-8'))
        
    return augmented_b64_images

@router.post("/encode-attendee/")
async def encode_attendee(request: EncodeAttendeeRequest):
    """
    Receives 3 base64 images of an attendee (front, left profile, right profile).
    Generates 2 augmented versions for each image (sharpness, brightness adjustment) non-blockingly.
    Sends all 9 images to inference API and returns the encodings to be used for sorting.
    """
    if len(request.attendee_images_base64) != 3:
        raise HTTPException(status_code=400, detail="Must provide exactly 3 attendee images (front, left, right).")
        
    try:
        # Offload CPU-heavy image manipulation to a separate thread
        import asyncio
        augmented_b64_images = await asyncio.to_thread(process_and_augment_images, request.attendee_images_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process and augment images: {e}")
        
    # Send all 9 images to inference API
    try:
        results = await call_inference_api(augmented_b64_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API Failed: {e}")
        
    embeddings_list = []
    for image_faces in results:
        # Assuming there is exactly 1 face per reference image setup
        if len(image_faces) == 1:
            embeddings_list.append(image_faces[0]["embedding"])
            
    if not embeddings_list:
        raise HTTPException(status_code=400, detail="Could not detect clear faces in the provided and augmented reference images.")

    return {
        "message": f"Successfully generated {len(embeddings_list)} encodings from 3 reference images.",
        "encodings": embeddings_list
    }

@router.post("/sort-attendee/")
async def sort_event_attendee(request: SortAttendeeRequest, db: AsyncSession = Depends(get_db)):
    """
    The 'Magic' Matcher algorithm:
    1. Receives 9 exact embeddings from the client.
    2. Averages them into a "Master Embedding".
    3. Searches the specific Event pgvector table for similar embeddings (< 0.4 distance).
    4. Returns strictly matched MinIO image paths.
    """
    if len(request.attendee_encodings) == 0:
        raise HTTPException(status_code=400, detail="Must provide at least one attendee encoding.")
        
    # Formatting vectors for Postgres literal query
    # Assign an ID to each of the 9 encodings so we can GROUP and COUNT them
    formatted_encodings = [f"'{str(emb)}'" for emb in request.attendee_encodings]
    values_clause = ", ".join([f"({i+1}, {emb}::vector)" for i, emb in enumerate(formatted_encodings)])
    
    # 3. Reference Dynamic Table explicitly
    EventModel = create_event_table_model(request.minio_folder_path)
    table_name = EventModel.__tablename__
    
    # Quick sanity check: Does this table exist?
    from sqlalchemy import inspect
    # We MUST use `run_sync` to perform synchronous inspection across an AsyncEngine
    async with engine.connect() as conn:
        def check_table(sync_conn):
            return inspect(sync_conn).has_table(table_name)
            
        has_table = await conn.run_sync(check_table)
        
    if not has_table:
         raise HTTPException(status_code=404, detail=f"No encoded data found for event {request.minio_folder_path}.")
         
    # 4. Search using pgvector Cosine similarity `<=>` operator. 
    # Distances < 0.4 generally correspond to high facial certainty threshold.
    SIMILARITY_THRESHOLD = 0.4
    
    # Optimized raw SQL string grouping all 9 encodings into a Common Table Expression
    # We evaluate every event photo against every reference profile.
    # K-NN Logic: An event photo MUST match at least 3 of the 9 attendee profiles to be valid.
    query_str = f"""
        WITH ref_encodings(id, embedding) AS (
            VALUES {values_clause}
        )
        SELECT p.image_path, COUNT(e.id) as match_count, MIN(p.embedding <=> e.embedding) as best_distance
        FROM {table_name} p
        CROSS JOIN ref_encodings e
        WHERE p.embedding <=> e.embedding < :threshold
        GROUP BY p.image_path
        HAVING COUNT(e.id) >= 3
        ORDER BY match_count DESC, best_distance ASC
    """
    query = text(query_str)
    
    result = await db.execute(query, {
        "threshold": SIMILARITY_THRESHOLD
    })
    
    matched_paths = [row[0] for row in result.all()]
    
    return {
        "event": request.minio_folder_path,
        "matches_found": len(matched_paths),
        "photos": matched_paths
    }
