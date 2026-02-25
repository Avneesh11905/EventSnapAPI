from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import numpy as np

from database import get_db, engine
from utils import call_inference_api
from tasks import create_event_table_model # Re-use the schema gen

router = APIRouter()

class SortAttendeeRequest(BaseModel):
    minio_folder_path: str
    attendee_images_base64: list[str]

@router.post("/sort-attendee/")
async def sort_event_attendee(request: SortAttendeeRequest, db: AsyncSession = Depends(get_db)):
    """
    The 'Magic' Matcher algorithm:
    1. Sends 3 raw Base64 strings to Inference API to get 3 exact embeddings.
    2. Averages them into a "Master Embedding".
    3. Searches the specific Event pgvector table for similar embeddings (< 0.4 distance).
    4. Returns strictly matched MinIO image paths.
    """
    if len(request.attendee_images_base64) == 0:
        raise HTTPException(status_code=400, detail="Must provide at least one attendee image.")
        
    # 1. Fetch live embeddings for the provided reference images
    try:
        results = await call_inference_api(request.attendee_images_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API Failed: {e}")
        
    embeddings_list = []
    for image_faces in results:
        # Assuming there is exactly 1 face per reference image setup
        if len(image_faces) == 1:
            embeddings_list.append(image_faces[0]["embedding"])
            
    if not embeddings_list:
        raise HTTPException(status_code=400, detail="Could not detect a clear face in the provided reference images.")
        
    # 2. Average embeddings to create a single Master vector
    emb_array = np.array(embeddings_list)
    master_embedding = np.mean(emb_array, axis=0)
    master_embedding = master_embedding / np.linalg.norm(master_embedding) # L2 normalize
    
    # Format vector for Postgres literal query
    master_embedding_list = master_embedding.tolist()
    
    # 3. Reference Dynamic Table explicitly
    EventModel = create_event_table_model(request.minio_folder_path)
    table_name = EventModel.__tablename__
    
    # Quick sanity check: Does this table exist?
    from sqlalchemy import inspect
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
         raise HTTPException(status_code=404, detail=f"No encoded data found for event {request.minio_folder_path}.")
         
    # 4. Search using pgvector Cosine similarity `<=>` operator. 
    # Distances < 0.4 generally correspond to high facial certainty threshold.
    SIMILARITY_THRESHOLD = 0.4
    
    # Optimized raw SQL string for vector similarity leveraging pgvector index
    # We select DISTINCT to avoid returning the same event photo twice if 
    # the matching face is somehow detected twice in the same image.
    query = text(f"""
        SELECT DISTINCT image_path 
        FROM {table_name}
        WHERE embedding <=> :master_embedding < :threshold
        ORDER BY embedding <=> :master_embedding ASC
    """)
    
    # We must explicitly cast the list to a string recognizable by Postgres `vector`
    result = await db.execute(query, {
        "master_embedding": str(master_embedding_list), 
        "threshold": SIMILARITY_THRESHOLD
    })
    
    matched_paths = [row[0] for row in result.all()]
    
    return {
        "event": request.minio_folder_path,
        "matches_found": len(matched_paths),
        "photos": matched_paths
    }
