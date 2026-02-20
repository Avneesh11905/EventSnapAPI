import base64
import httpx
import io
from PIL import Image
from aioboto3 import Session
from botocore.config import Config

from config import settings

def get_minio_session():
    return Session(
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
    )

async def list_minio_images(folder_path: str) -> list[str]:
    """Retrieve a list of all object keys inside a MinIO folder."""
    session = get_minio_session()
    keys = []
    
    # Ensure trailing slash for proper prefix matching
    prefix = folder_path if folder_path.endswith('/') else f"{folder_path}/"
    
    async with session.client(
        's3', 
        endpoint_url=settings.MINIO_ENDPOINT,
        config=Config(signature_version='s3v4')
    ) as s3:
        paginator = s3.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=settings.MINIO_BUCKET_NAME, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    keys.append(obj['Key'])
                    
    return keys

def resize_image_bytes(image_data: bytes, max_size: int) -> str:
    """Synchronous CPU-bound PIL operation to be run in a background thread."""
    with Image.open(io.BytesIO(image_data)) as img:
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        optimized_data = buffer.getvalue()
        
    return base64.b64encode(optimized_data).decode('utf-8')

async def download_minio_image_b64(s3_client, key: str, max_size: int = 1024) -> str | None:
    """Download single image, resize it, and return as a Base64 string."""
    try:
        response = await s3_client.get_object(Bucket=settings.MINIO_BUCKET_NAME, Key=key)
        image_data = await response['Body'].read()
        
        import asyncio
        # Offload the resizing to a threadpool so it doesn't block the async loop
        b64_str = await asyncio.to_thread(resize_image_bytes, image_data, max_size)
            
        return b64_str
    except Exception as e:
        print(f"Error downloading/resizing {key}: {e}")
        return None

async def call_inference_api(
    b64_images: list[str], 
    max_faces: int = 0,
    detection_conf: float = 0.5,
    nms_threshold: float = 0.4
) -> list[dict]:
    """Send batch of Base64 images and detection parameters to the HF Inference API."""
    headers = {"Content-Type": "application/json"}
    if settings.INFERENCE_API_TOKEN:
        headers["Authorization"] = f"Bearer {settings.INFERENCE_API_TOKEN}"
        
    payload = {
        "inputs": b64_images,
        "parameters": {
            "max_faces": max_faces,  # 0 = All faces
            "detection_conf": detection_conf,
            "nms_threshold": nms_threshold
        }
    }
    
    # Fast HTTPX client for async REST requests
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.INFERENCE_API_URL}/", 
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        
        # Result format: {"batch_faces": [ [face1, face2], [face1], [] ]}
        data = response.json()
        if "error" in data:
             raise RuntimeError(f"Inference API Error: {data['error']}")
             
        return data.get("batch_faces", [])
