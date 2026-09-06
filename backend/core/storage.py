import asyncio
import logging
from functools import cached_property
from typing import Any

from backend.core.config import settings

logger = logging.getLogger(__name__)


class StorageNotConfiguredError(RuntimeError):
    pass


class ObjectStorageError(RuntimeError):
    pass


AVATAR_CACHE_CONTROL = "public, max-age=31536000, immutable"
PRIVATE_UPLOAD_CACHE_CONTROL = "private, max-age=3600"


class ObjectStorage:
    @property
    def is_configured(self) -> bool:
        return bool(settings.STORAGE_BUCKET)

    @cached_property
    def _client(self):
        try:
            import boto3
            from botocore.client import Config
        except ImportError as exc:
            raise ObjectStorageError(
                "Object storage dependencies are not installed. Run `uv sync` in `backend/`."
            ) from exc

        session = boto3.session.Session()
        return session.client(
            "s3",
            region_name=settings.STORAGE_REGION,
            endpoint_url=settings.STORAGE_ENDPOINT_URL or None,
            aws_access_key_id=settings.STORAGE_ACCESS_KEY or None,
            aws_secret_access_key=settings.STORAGE_SECRET_KEY or None,
            use_ssl=settings.STORAGE_USE_SSL,
            config=Config(
                signature_version="s3v4",
                connect_timeout=2,
                read_timeout=2,
                retries={"max_attempts": 1, "mode": "standard"},
                s3={"addressing_style": "path" if settings.STORAGE_FORCE_PATH_STYLE else "auto"},
            ),
        )

    async def ensure_bucket(self) -> None:
        if not self.is_configured or not settings.STORAGE_AUTO_CREATE_BUCKET:
            return

        def _ensure_bucket() -> None:
            try:
                self._client.head_bucket(Bucket=settings.STORAGE_BUCKET)
            except Exception:
                create_kwargs = {"Bucket": settings.STORAGE_BUCKET}
                if settings.STORAGE_REGION != "us-east-1":
                    create_kwargs["CreateBucketConfiguration"] = {
                        "LocationConstraint": settings.STORAGE_REGION
                    }
                self._client.create_bucket(**create_kwargs)


        try:
            await asyncio.to_thread(_ensure_bucket)
        except Exception as exc:
            logger.warning("failed to ensure storage bucket %s: %s", settings.STORAGE_BUCKET, exc)

    async def upload_bytes(
        self,
        *,
        object_key: str,
        body: bytes,
        content_type: str,
        cache_control: str = AVATAR_CACHE_CONTROL,
    ) -> str:
        if not self.is_configured:
            raise StorageNotConfiguredError(
                "Object storage is not configured. Set STORAGE_BUCKET and storage credentials."
            )

        def _upload() -> None:
            put_kwargs: dict[str, str] = {
                "Bucket": settings.STORAGE_BUCKET,
                "Key": object_key,
                "Body": body,
                "ContentType": content_type,
            }
            if cache_control:
                put_kwargs["CacheControl"] = cache_control
            self._client.put_object(**put_kwargs)

        try:
            await asyncio.to_thread(_upload)
        except Exception as exc:
            raise ObjectStorageError("Failed to upload avatar to object storage") from exc

        return self.public_url_for(object_key)

    async def download_bytes(self, object_key: str) -> bytes:
        if not self.is_configured:
            raise StorageNotConfiguredError(
                "Object storage is not configured. Set STORAGE_BUCKET and storage credentials."
            )

        def _download() -> bytes:
            response = self._client.get_object(Bucket=settings.STORAGE_BUCKET, Key=object_key)
            return response["Body"].read()

        try:
            return await asyncio.to_thread(_download)
        except Exception as exc:
            raise ObjectStorageError(f"Failed to download object {object_key}") from exc

    async def presigned_download_url(self, object_key: str, *, expires_in: int = 300) -> str:
        if not self.is_configured:
            raise StorageNotConfiguredError(
                "Object storage is not configured. Set STORAGE_BUCKET and storage credentials."
            )

        def _presign() -> str:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.STORAGE_BUCKET, "Key": object_key},
                ExpiresIn=expires_in,
            )

        try:
            return await asyncio.to_thread(_presign)
        except Exception as exc:
            raise ObjectStorageError("Failed to authorize object download") from exc

    def download_bytes_sync(self, object_key: str, *, bucket: str | None = None) -> bytes:
        if not self.is_configured and not bucket:
            raise StorageNotConfiguredError(
                "Object storage is not configured. Set STORAGE_BUCKET and storage credentials."
            )
        response = self._client.get_object(
            Bucket=bucket or settings.STORAGE_BUCKET, Key=object_key
        )
        return response["Body"].read()

    def upload_bytes_sync(
        self,
        *,
        object_key: str,
        body: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> None:
        if not self.is_configured:
            raise StorageNotConfiguredError(
                "Object storage is not configured. Set STORAGE_BUCKET and storage credentials."
            )
        put_kwargs: dict[str, Any] = {
            "Bucket": settings.STORAGE_BUCKET,
            "Key": object_key,
            "Body": body,
            "ContentType": content_type,
        }
        if metadata:
            put_kwargs["Metadata"] = metadata
        self._client.put_object(**put_kwargs)

    def delete_object_sync(self, object_key: str, *, bucket: str | None = None) -> None:
        if not self.is_configured and not bucket:
            return
        self._client.delete_object(Bucket=bucket or settings.STORAGE_BUCKET, Key=object_key)

    async def delete_object(self, object_key: str | None) -> None:
        if not self.is_configured or not object_key:
            return

        def _delete() -> None:
            self._client.delete_object(Bucket=settings.STORAGE_BUCKET, Key=object_key)

        try:
            await asyncio.to_thread(_delete)
        except Exception as exc:
            logger.warning("failed to delete storage object %s: %s", object_key, exc)

    def public_url_for(self, object_key: str) -> str:
        if settings.STORAGE_PUBLIC_BASE_URL:
            return f"{settings.STORAGE_PUBLIC_BASE_URL.rstrip('/')}/{object_key}"

        if settings.STORAGE_ENDPOINT_URL:
            base = settings.STORAGE_ENDPOINT_URL.rstrip("/")
            return f"{base}/{settings.STORAGE_BUCKET}/{object_key}"

        if settings.STORAGE_REGION == "us-east-1":
            return f"https://{settings.STORAGE_BUCKET}.s3.amazonaws.com/{object_key}"

        return (
            f"https://{settings.STORAGE_BUCKET}.s3.{settings.STORAGE_REGION}.amazonaws.com/"
            f"{object_key}"
        )


object_storage = ObjectStorage()
