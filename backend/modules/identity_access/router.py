import logging

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps.auth import get_authenticated_user
from backend.api.deps.db import get_db
from backend.core.config import settings
from backend.core.rate_limit import (
    auth_rate_limit_key,
    check_rate_limit,
    clear_rate_limit,
    enforce_rate_limit,
    increment_rate_limit,
)
from backend.core.security import create_access_token, generate_csrf_token
from backend.modules.identity_access.models import User
from backend.modules.identity_access.schemas import (
    AuthSessionResponse,
    AuthUserResponse,
    ForgotPasswordRequest,
    GenericMessageResponse,
    MfaDisableRequest,
    MfaEnableResponse,
    MfaVerifyRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    SignInRequest,
    SignUpRequest,
    VerifyEmailRequest,
)
from backend.modules.identity_access.service import IdentityService

router = APIRouter()
logger = logging.getLogger("backend.auth")

_REFRESH_COOKIE_MAX_AGE = 60 * 60 * 24 * settings.REFRESH_TOKEN_EXPIRE_DAYS
_ACCESS_COOKIE_MAX_AGE = 60 * settings.ACCESS_TOKEN_EXPIRE_MINUTES


def _cookie_kwargs() -> dict:
    return {
        "secure": settings.COOKIE_SECURE,
        "samesite": settings.COOKIE_SAMESITE,
        "domain": settings.COOKIE_DOMAIN,
    }


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=_REFRESH_COOKIE_MAX_AGE,
        path="/api/v1/auth",
        **_cookie_kwargs(),
    )


def _set_access_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.ACCESS_COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=_ACCESS_COOKIE_MAX_AGE,
        path="/",
        **_cookie_kwargs(),
    )


def _set_csrf_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.CSRF_COOKIE_NAME,
        value=token,
        httponly=False,
        max_age=_REFRESH_COOKIE_MAX_AGE,
        path="/",
        **_cookie_kwargs(),
    )


def _clear_auth_cookies(response: Response) -> None:
    for key, path in (
        (settings.ACCESS_COOKIE_NAME, "/"),
        (settings.REFRESH_COOKIE_NAME, "/api/v1/auth"),
        (settings.CSRF_COOKIE_NAME, "/"),
    ):
        response.delete_cookie(key, path=path, domain=settings.COOKIE_DOMAIN)


def _build_user(user: User) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_verified=user.is_verified,
        is_admin=user.is_admin,
        mfa_enabled=user.mfa_enabled,
    )


# ------------------------------------------------------------------ core auth

@router.post("/sign-up", response_model=GenericMessageResponse, status_code=202)
async def sign_up(
    payload: SignUpRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    await check_rate_limit(
        key=(
            f"rate_limit:signup:"
            f"{request.client.host if request.client else 'unknown'}:{payload.email}"
        ),
        max_attempts=5,
        window_seconds=3600,
    )
    service = IdentityService(db)
    await service.sign_up(
        payload.email,
        payload.password,
        payload.full_name,
        payload.admin_invite_code,
    )
    detail = (
        "If the account can be created, a verification email will be sent shortly."
        if settings.REQUIRE_EMAIL_VERIFICATION
        else "If the account can be created, you can sign in shortly."
    )
    return GenericMessageResponse(detail=detail)


@router.post("/sign-in", response_model=AuthSessionResponse)
async def sign_in(
    payload: SignInRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    await check_rate_limit(
        key=auth_rate_limit_key(request, payload.email),
        max_attempts=10,
        window_seconds=60,
    )
    failure_key = (
        f"rate_limit:auth_fail:"
        f"{request.client.host if request.client else 'unknown'}:{payload.email}"
    )
    await enforce_rate_limit(failure_key, settings.AUTH_FAILURE_LIMIT)
    service = IdentityService(db)
    try:
        result = await service.sign_in(payload.email, payload.password, payload.mfa_code)
    except HTTPException as exc:
        if exc.status_code in {401, 403}:
            await increment_rate_limit(failure_key, settings.AUTH_FAILURE_WINDOW_SECONDS)
            logger.warning(
                "authentication_failed email=%s ip=%s status=%s",
                payload.email,
                request.client.host if request.client else "unknown",
                exc.status_code,
            )
        raise

    await clear_rate_limit(failure_key)
    access_token = create_access_token(result["user"].id, result["session_id"])
    csrf_token = generate_csrf_token()
    _set_access_cookie(response, access_token)
    _set_refresh_cookie(response, result["refresh_token"])
    _set_csrf_cookie(response, csrf_token)
    return AuthSessionResponse(user=_build_user(result["user"]))


@router.post("/refresh", response_model=AuthSessionResponse)
async def refresh(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=settings.REFRESH_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
):
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    service = IdentityService(db)
    result = await service.refresh(refresh_token)
    access_token = create_access_token(result["user"].id, result["session_id"])
    csrf_token = generate_csrf_token()
    _set_access_cookie(response, access_token)
    _set_refresh_cookie(response, result["refresh_token"])
    _set_csrf_cookie(response, csrf_token)
    return AuthSessionResponse(user=_build_user(result["user"]))


@router.post("/logout", status_code=204)
async def logout(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=settings.REFRESH_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
):
    if refresh_token:
        service = IdentityService(db)
        await service.logout(refresh_token)
    _clear_auth_cookies(response)


@router.get("/me", response_model=AuthUserResponse)
async def me(current_user: User = Depends(get_authenticated_user)):
    return _build_user(current_user)


# ------------------------------------------------------------------ email verification

@router.post("/verify-email", status_code=204)
async def verify_email(
    payload: VerifyEmailRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    await check_rate_limit(
        key=f"rate_limit:verify:{request.client.host if request.client else 'unknown'}",
        max_attempts=10,
        window_seconds=3600,
    )
    service = IdentityService(db)
    await service.verify_email(payload.token)


@router.post("/resend-verification", status_code=204)
async def resend_verification(
    payload: ResendVerificationRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    await check_rate_limit(
        key=f"rate_limit:resend:{request.client.host if request.client else 'unknown'}",
        max_attempts=3,
        window_seconds=300,
    )
    service = IdentityService(db)
    await service.resend_verification(payload.email)


# ------------------------------------------------------------------ password reset

@router.post("/forgot-password", status_code=204)
async def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    await check_rate_limit(
        key=(
            f"rate_limit:forgot:"
            f"{request.client.host if request.client else 'unknown'}:{payload.email}"
        ),
        max_attempts=5,
        window_seconds=300,
    )
    service = IdentityService(db)
    await service.forgot_password(payload.email)


@router.post("/reset-password", status_code=204)
async def reset_password(payload: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    service = IdentityService(db)
    await service.reset_password(payload.token, payload.new_password)


# ------------------------------------------------------------------ MFA

@router.post("/mfa/enable", response_model=MfaEnableResponse)
async def mfa_enable(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_authenticated_user),
):
    service = IdentityService(db)
    return await service.mfa_enable(current_user)


@router.post("/mfa/verify", status_code=204)
async def mfa_verify(
    payload: MfaVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_authenticated_user),
):
    service = IdentityService(db)
    await service.mfa_verify_enable(current_user, payload.code)


@router.post("/mfa/disable", status_code=204)
async def mfa_disable(
    payload: MfaDisableRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_authenticated_user),
):
    service = IdentityService(db)
    await service.mfa_disable(current_user, payload.code)
