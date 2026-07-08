from application.ports.storage import StorageService
import asyncio
from aioboto3 import Session
from botocore.config import Config
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List
import zipfile
import tempfile
import os
import base64
import io
from PIL import Image

class MinioStorageService(StorageService):
    def __init__(self, endpoint_url: str, bucket_name: str, access_key: str, secret_key: str):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key

    def _get_session(self):
        return Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

    async def list_images(self, folder_path: str) -> List[str]:
        session = self._get_session()
        keys = []
        prefix = folder_path if folder_path.endswith('/') else f"{folder_path}/"
        async with session.client('s3', endpoint_url=self.endpoint_url, config=Config(signature_version='s3v4')) as s3:
            paginator = s3.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        keys.append(obj['Key'])
        return keys

    @staticmethod
    def resize_image_bytes(image_data: bytes, max_size: int) -> str:
        with Image.open(io.BytesIO(image_data)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            optimized_data = buffer.getvalue()
        return base64.b64encode(optimized_data).decode('utf-8')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ClientError, asyncio.TimeoutError, ConnectionError)),
        reraise=True
    )
    async def download_image_b64(self, key: str, max_size: int = 1024) -> str | None:
        session = self._get_session()
        async with session.client('s3', endpoint_url=self.endpoint_url, config=Config(signature_version='s3v4', max_pool_connections=64)) as s3_client:
            response = await s3_client.get_object(Bucket=self.bucket_name, Key=key)
            image_data = await response['Body'].read()
            
        b64_str = await asyncio.to_thread(self.resize_image_bytes, image_data, max_size)
        return b64_str

    async def create_zip_from_images(self, zip_path: str, image_paths: List[dict], progress_callback=None) -> None:
        session = self._get_session()
        total = len(image_paths)
        s3_config = Config(max_pool_connections=10, retries={'max_attempts': 0})
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name
            
            async with session.client('s3', endpoint_url=self.endpoint_url, config=s3_config) as s3_client:
                with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, img in enumerate(image_paths):
                        filename = img['filename']
                        key = img['path']
                        
                        max_retries = 3
                        for attempt in range(max_retries + 1):
                            try:
                                response = await s3_client.get_object(Bucket=self.bucket_name, Key=key)
                                data = await response['Body'].read()
                                zf.writestr(filename, data)
                                break
                            except Exception as e:
                                if attempt < max_retries:
                                    wait = (2 ** attempt) + 1
                                    if progress_callback:
                                        progress_callback(i, 'RETRYING', f'Retry {attempt+1}/{max_retries} for {filename}...')
                                    await asyncio.sleep(wait)
                                else:
                                    print(f"Permanent failure for {key} after {max_retries} retries: {e}")
                        
                        if progress_callback:
                            progress_callback(i + 1, 'PROCESSING', f'Downloaded {i+1}/{total} images')
                
                if progress_callback:
                    progress_callback(total, 'UPLOADING', 'Uploading ZIP to storage...')
                    
                with open(tmp_path, 'rb') as f:
                    await s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=zip_path,
                        Body=f,
                        ContentType='application/zip'
                    )
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def check_zip_exists(self, zip_key: str) -> bool:
        session = self._get_session()
        async with session.client('s3', endpoint_url=self.endpoint_url) as s3:
            try:
                await s3.head_object(Bucket=self.bucket_name, Key=zip_key)
                return True
            except Exception:
                return False
