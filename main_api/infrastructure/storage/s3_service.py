from config import settings
from application.ports.storage import IStorageService
import asyncio
from aioboto3 import Session
from botocore.config import Config
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from typing import List
import zipfile
import tempfile
from domain.exceptions import StorageDownloadError
import os
import base64


class S3StorageService(IStorageService):
    def __init__(
        self, endpoint_url: str, bucket_name: str, access_key: str, secret_key: str
    ):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key

    def _get_session(self):
        return Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="auto",
        )

    async def list_images(self, folder_path: str) -> List[str]:
        session = self._get_session()
        keys = []
        prefix = folder_path if folder_path.endswith("/") else f"{folder_path}/"
        async with session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            config=Config(
                signature_version="s3v4",
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        ) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix
            ):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
        return keys

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (ClientError, asyncio.TimeoutError, ConnectionError)
        ),
        reraise=True,
    )
    async def _download_images_base(
        self, keys: list[str], as_b64: bool
    ) -> list[str | bytes | Exception]:
        session = self._get_session()

        async def fetch_single(client, k):
            try:
                res = await client.get_object(Bucket=self.bucket_name, Key=k)
                data = await res["Body"].read()
                return base64.b64encode(data).decode("utf-8") if as_b64 else data
            except Exception as e:
                return e

        async with session.client(
            "s3",
            endpoint_url=self.endpoint_url,
            config=Config(
                signature_version="s3v4",
                max_pool_connections=settings.S3_MAX_POOL_CONNECTIONS,
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        ) as s3_client:
            tasks = [fetch_single(s3_client, key) for key in keys]
            results = await asyncio.gather(*tasks)
            return results


class S3StorageServiceB64(S3StorageService):
    async def download_images(self, keys: list[str]) -> list[str | Exception]:
        return await self._download_images_base(keys, as_b64=True)  # type: ignore


class S3StorageServiceBytes(S3StorageService):
    async def download_images(self, keys: list[str]) -> list[bytes | Exception]:
        return await self._download_images_base(keys, as_b64=False)  # type: ignore

    async def create_zip_from_images(
        self, zip_path: str, image_paths: List[dict], progress_callback=None
    ) -> None:
        session = self._get_session()
        total = len(image_paths)
        s3_config = Config(
            signature_version="s3v4",
            max_pool_connections=settings.S3_MAX_POOL_CONNECTIONS,
            retries={"max_attempts": 0},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name

            async with session.client(
                "s3", endpoint_url=self.endpoint_url, config=s3_config
            ) as s3_client:
                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, img in enumerate(image_paths):
                        filename = img["filename"]
                        key = img["path"]

                        max_retries = 3
                        for attempt in range(max_retries + 1):
                            try:
                                response = await s3_client.get_object(
                                    Bucket=self.bucket_name, Key=key
                                )
                                data = await response["Body"].read()
                                zf.writestr(filename, data)
                                break
                            except Exception as e:
                                if attempt < max_retries:
                                    wait = (2**attempt) + 1
                                    if progress_callback:
                                        progress_callback(
                                            i,
                                            "RETRYING",
                                            f"Retry {attempt + 1}/{max_retries} for {filename}...",
                                        )
                                    await asyncio.sleep(wait)
                                else:
                                    raise StorageDownloadError(
                                        f"Permanent failure for {key} after {max_retries} retries: {e}"
                                    )

                        if progress_callback:
                            progress_callback(
                                i + 1,
                                "PROCESSING",
                                f"Downloaded {i + 1}/{total} images",
                            )

                if progress_callback:
                    progress_callback(total, "UPLOADING", "Uploading ZIP to storage...")

                with open(tmp_path, "rb") as f:
                    file_data = f.read()

                await s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=zip_path,
                    Body=file_data,
                    ContentType="application/zip",
                )

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def check_zip_exists(self, zip_key: str) -> bool:
        session = self._get_session()
        async with session.client("s3", endpoint_url=self.endpoint_url) as s3:
            try:
                await s3.head_object(Bucket=self.bucket_name, Key=zip_key)
                return True
            except Exception:
                return False

    async def delete_folder(self, prefix: str) -> None:
        session = self._get_session()
        async with session.client("s3", endpoint_url=self.endpoint_url) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self.bucket_name, Prefix=prefix
            ):
                if "Contents" in page:
                    objects_to_delete = [
                        {"Key": obj["Key"]} for obj in page["Contents"]
                    ]
                    if objects_to_delete:
                        await s3.delete_objects(
                            Bucket=self.bucket_name,
                            Delete={"Objects": objects_to_delete, "Quiet": True},
                        )

    async def delete_objects(self, keys: List[str]) -> None:
        if not keys:
            return
        session = self._get_session()
        async with session.client("s3", endpoint_url=self.endpoint_url) as s3:
            objects_to_delete = [{"Key": k} for k in keys]
            await s3.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": objects_to_delete, "Quiet": True},
            )
