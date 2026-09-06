import mimetypes
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_current_user
from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.core.storage import ObjectStorageError, StorageNotConfiguredError, object_storage
from backend.modules.identity_access.models import User
from backend.modules.profile.schemas import ProfileResponse, ProfileUpdate
from backend.modules.profile.serializers import profile_to_response
from backend.modules.profile.service import ProfileService

router = APIRouter()


def _build_avatar_object_key(user_id: str, filename: str | None, content_type: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not suffix:
        suffix = mimetypes.guess_extension(content_type) or ".bin"
    return f"avatars/{user_id}/{uuid4().hex}{suffix}"


@router.get("", response_model=ProfileResponse)
async def get_profile(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProfileService(db)
    return await service.get_profile_response(current_user.id)


@router.put("", response_model=ProfileResponse)
async def update_profile(
    payload: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProfileService(db)
    profile = await service.update_profile(
        current_user.id, payload.bio, payload.location, payload.website
    )
    return profile_to_response(profile)


@router.post("/avatar", response_model=ProfileResponse)
async def upload_avatar(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Avatar upload only supports image files")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded avatar file is empty")
    if len(payload) > settings.STORAGE_AVATAR_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Avatar exceeds the maximum size of {settings.STORAGE_AVATAR_MAX_BYTES} bytes",
        )

    object_key = _build_avatar_object_key(current_user.id, file.filename, file.content_type)
    try:
        avatar_url = await object_storage.upload_bytes(
            object_key=object_key,
            body=payload,
            content_type=file.content_type,
        )
    except StorageNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ObjectStorageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    service = ProfileService(db)
    try:
        profile, previous_key = await service.replace_avatar(
            current_user.id,
            avatar_url=avatar_url,
            storage_key=object_key,
        )
    except Exception:
        await object_storage.delete_object(object_key)
        raise

    if previous_key and previous_key != object_key:
        await object_storage.delete_object(previous_key)
    return profile_to_response(profile)


@router.delete("/avatar", status_code=204)
async def delete_avatar(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = ProfileService(db)
    previous_key = await service.clear_avatar(current_user.id)
    await object_storage.delete_object(previous_key)
