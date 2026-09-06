from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.core.schemas import RequestModel


class SignUpRequest(RequestModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None
    admin_invite_code: str | None = None


class SignInRequest(RequestModel):
    email: EmailStr
    password: str
    mfa_code: str | None = Field(default=None, min_length=6, max_length=6)

    @field_validator("mfa_code", mode="before")
    @classmethod
    def normalize_blank_mfa_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return value


class AuthUserResponse(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None
    is_verified: bool
    is_admin: bool = False
    mfa_enabled: bool = False


class AuthSessionResponse(BaseModel):
    user: AuthUserResponse


class GenericMessageResponse(BaseModel):
    detail: str


# Email verification
class VerifyEmailRequest(RequestModel):
    token: str


class ResendVerificationRequest(RequestModel):
    email: EmailStr


# Password reset
class ForgotPasswordRequest(RequestModel):
    email: EmailStr


class ResetPasswordRequest(RequestModel):
    token: str
    new_password: str = Field(min_length=8)


# MFA
class MfaEnableResponse(BaseModel):
    secret: str
    provisioning_uri: str


class MfaVerifyRequest(RequestModel):
    code: str = Field(min_length=6, max_length=6)


class MfaDisableRequest(RequestModel):
    code: str = Field(min_length=6, max_length=6)
