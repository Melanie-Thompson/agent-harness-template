import secrets

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from auth.email import send_email
from config import Config
from db.sessions import get_dragonfly
from logging_decorator import log_call

_serializer: URLSafeTimedSerializer | None = None


def _get_serializer() -> URLSafeTimedSerializer:
    global _serializer
    if _serializer is None:
        _serializer = URLSafeTimedSerializer(Config.MAGIC_LINK_SECRET)
    return _serializer


@log_call
async def send_magic_link(email: str) -> None:
    token = _get_serializer().dumps(email, salt="magic-link")
    link = f"{Config.APP_URL}/auth/magic/verify?token={token}"
    await send_email(
        to=email,
        subject="Your login link",
        html=f'<p>Click <a href="{link}">here</a> to log in. This link expires in {Config.MAGIC_LINK_EXPIRE // 60} minutes.</p>',
    )


def verify_magic_link(token: str) -> str | None:
    try:
        return _get_serializer().loads(token, salt="magic-link", max_age=Config.MAGIC_LINK_EXPIRE)
    except (BadSignature, SignatureExpired):
        return None


@log_call
async def send_otp(email: str) -> None:
    code = f"{secrets.randbelow(1000000):06d}"
    r = get_dragonfly()
    await r.set(f"otp:{email}", code, ex=Config.OTP_EXPIRE)
    await send_email(
        to=email,
        subject="Your login code",
        html=f"<p>Your verification code is: <strong>{code}</strong></p><p>It expires in {Config.OTP_EXPIRE // 60} minutes.</p>",
    )


@log_call
async def verify_otp(email: str, code: str) -> bool:
    r = get_dragonfly()
    stored = await r.get(f"otp:{email}")
    if not stored or stored != code:
        return False
    await r.delete(f"otp:{email}")
    return True