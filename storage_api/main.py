import urllib3.util.ssl_match_hostname
import io
import mimetypes
import zipfile
import aioboto3
import botocore.config
import secrets
import string
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from contextlib import asynccontextmanager

from config import settings

# Initialize aioboto3 session
session = aioboto3.Session(
    aws_access_key_id=settings.MINIO_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_SECRET_KEY,
)

async def check_or_create_bucket():
    """Ensure the MinIO bucket exists on startup."""
    async with session.client(
        's3',
        endpoint_url=settings.MINIO_ENDPOINT,
        aws_session_token=None,
        config=botocore.config.Config(signature_version='s3v4'),
    ) as s3_client:
        try:
            # Check if bucket exists
            await s3_client.head_bucket(Bucket=settings.MINIO_BUCKET_NAME)
            print(f"Bucket '{settings.MINIO_BUCKET_NAME}' already exists.")
        except Exception as e:
            # If it doesn't exist, we try to create it
            # The exception varies, normally ClientError (404 Not Found)
            print(f"Bucket '{settings.MINIO_BUCKET_NAME}' does not exist, creating...")
            try:
                await s3_client.create_bucket(Bucket=settings.MINIO_BUCKET_NAME)
                print(f"Bucket '{settings.MINIO_BUCKET_NAME}' created successfully.")
            except Exception as create_exc:
                print(f"Failed to create bucket: {create_exc}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure MinIO bucket exists
    await check_or_create_bucket()
    yield
    # Shutdown logic if any

app = FastAPI(title="Eventsnap Storage API", lifespan=lifespan)

import aiofiles
import os
from fastapi import BackgroundTasks

@app.post("/upload-zip/")
async def upload_zip_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Handles robust uploading of massively large ZIP files (e.g. 5GB+).
    Instead of crashing the server's RAM, the file is streamed to disk in chunks,
    and then a background task is triggered to unzip and upload to MinIO.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a .zip archive."
        )

    folder_name = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    temp_zip_path = f"/tmp/{folder_name}.zip"

    # 1. Stream the multi-gigabyte upload to disk in chunks (Does not blow up RAM)
    async with aiofiles.open(temp_zip_path, 'wb') as out_file:
        while content := await file.read(1024 * 1024):  # Read in 1MB chunks
            await out_file.write(content)

    # 2. Trigger Background Task to handle the MinIO Extraction & Uploading
    background_tasks.add_task(process_huge_zip_background, temp_zip_path, folder_name)

    return {
        "message": "Files received successfully. Extraction and MinIO upload triggered in background.",
        "folder_name": folder_name,
        "status": "processing"
    }

@app.delete("/delete-bucket/{bucket_name}")
async def delete_bucket(bucket_name: str):
    """
    Deletes a MinIO bucket by name.
    Because S3/MinIO requires a bucket to be empty before deleting,
    this endpoint first deletes all objects inside the bucket.
    """
    async with session.client('s3', endpoint_url=settings.MINIO_ENDPOINT) as s3_client:
        try:
            # Check if bucket exists
            await s3_client.head_bucket(Bucket=bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' not found.")
            raise HTTPException(status_code=500, detail=str(e))
            
        try:
            # 1. Delete all objects in the bucket
            paginator = s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    objects_to_delete = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects_to_delete:
                        await s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
                        
            # 2. Delete the empty bucket
            await s3_client.delete_bucket(Bucket=bucket_name)
            return {"message": f"Bucket '{bucket_name}' and all its contents were successfully deleted."}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete bucket: {str(e)}")

async def process_huge_zip_background(zip_path: str, folder_name: str):
    """
    Background Task: Extracts images from the ZIP on disk and uploads to MinIO safely using 
    a Semaphore to limit concurrent S3 connections.
    """
    extracted_images = []
    
    # CPU-bound zip extraction to a separate thread
    def extract_images_from_zip_disk(path: str):
        extracted = []
        with zipfile.ZipFile(path, 'r') as zf:
            for file_info in zf.infolist():
                if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                    continue
                
                mime_type, _ = mimetypes.guess_type(file_info.filename)
                if mime_type and mime_type.startswith('image/'):
                    with zf.open(file_info) as img_file:
                        img_data = img_file.read()
                    
                    basename = file_info.filename.split('/')[-1]
                    extracted.append({
                        "basename": basename,
                        "mime_type": mime_type,
                        "data": img_data
                    })
        return extracted

    try:
        extracted_images = await asyncio.to_thread(extract_images_from_zip_disk, zip_path)
        
        # Safe async MinIO upload (Limit to 50 concurrent uploads so we don't DDoD ourselves)
        semaphore = asyncio.Semaphore(50)

        async def upload_image_safe(s3_client, item):
            async with semaphore:
                object_key = f"{folder_name}/{item['basename']}"
                await s3_client.put_object(
                    Bucket=settings.MINIO_BUCKET_NAME,
                    Key=object_key,
                    Body=item['data'],
                    ContentType=item['mime_type']
                )
                return object_key

        async with session.client('s3', endpoint_url=settings.MINIO_ENDPOINT) as s3_client:
            tasks = [upload_image_safe(s3_client, item) for item in extracted_images]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for index, res in enumerate(results):
                if isinstance(res, Exception):
                    print(f"Failed MinIO upload {extracted_images[index]['basename']}: {res}")
                else:
                    success_count += 1
            print(f"Background Upload Complete for {folder_name}: {success_count}/{len(extracted_images)} succeeded.")

    except Exception as e:
        print(f"Critical error processing zip in background: {e}")
    finally:
        # 3. Clean up the massive ZIP file from disk
        if os.path.exists(zip_path):
            os.remove(zip_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app='main:app', host="0.0.0.0", port=8000 , reload=True)