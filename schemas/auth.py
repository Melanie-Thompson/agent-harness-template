from pydantic import BaseModel, EmailStr, Field, field_validator


class MagicLinkRequest(BaseModel):
    email: EmailStr


class OTPSendRequest(BaseModel):
    email: EmailStr


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class AuthResponse(BaseModel):
    message: str
    user_id: str