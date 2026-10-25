from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse

from auth.oauth import oauth
from auth.passwordless import send_magic_link, send_otp, verify_magic_link, verify_otp
from config import Config
from errors import InvalidCredentialError, OAuthEmailError
from db.meta import build_meta_create
from db.mongo import get_mongo_db
from db.sessions import create_session
from schemas.auth import AuthResponse, MagicLinkRequest, OTPSendRequest, OTPVerifyRequest
from schemas.session import MessageResponse

router = APIRouter(prefix="/auth", tags=["auth"])

DEFAULT_PERMISSIONS = ["read:self"]


async def _authenticate_user(
    email: str,
    response: Response,
    first_name: str = "",
    last_name: str = "",
    auth_provider: str | None = None,
) -> dict:
    db = get_mongo_db()
    user = await db.users.find_one({"email": email, "_meta.deleted_at": None})

    if not user:
        doc = {
            "email": email,
            "first_name": first_name,
            "last_name": last_name,
            "middle_names": [],
            "auth_provider": auth_provider,
            "email_verified": True,
            **build_meta_create("system"),
        }
        result = await db.users.insert_one(doc)
        user_id = str(result.inserted_id)
    else:
        user_id = str(user["_id"])

    session_id = await create_session(user_id, DEFAULT_PERMISSIONS)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=Config.COOKIE_SECURE,
        samesite="lax",
        max_age=Config.SESSION_EXPIRE_SECONDS,
    )
    return {"user_id": user_id}


# --- OAuth ---

@router.get("/google", summary="Start Google OAuth flow")
async def google_login(request: Request):
    redirect_uri = f"{Config.APP_URL}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/google/callback", summary="Google OAuth callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    userinfo = token.get("userinfo", {})
    email = userinfo.get("email")
    if not email:
        raise OAuthEmailError("Google")

    response = RedirectResponse(url="/")
    await _authenticate_user(
        email=email,
        response=response,
        first_name=userinfo.get("given_name", ""),
        last_name=userinfo.get("family_name", ""),
        auth_provider="google",
    )
    return response


@router.get("/github", summary="Start GitHub OAuth flow")
async def github_login(request: Request):
    redirect_uri = f"{Config.APP_URL}/auth/github/callback"
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/github/callback", summary="GitHub OAuth callback")
async def github_callback(request: Request):
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get("user", token=token)
    profile = resp.json()

    email = profile.get("email")
    if not email:
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        primary = next((e for e in emails if e.get("primary")), None)
        email = primary["email"] if primary else None

    if not email:
        raise OAuthEmailError("GitHub")

    name_parts = (profile.get("name") or "").split(" ", 1)
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    response = RedirectResponse(url="/")
    await _authenticate_user(
        email=email,
        response=response,
        first_name=first_name,
        last_name=last_name,
        auth_provider="github",
    )
    return response


# --- Passwordless: Magic Link ---

@router.post("/magic-link", response_model=MessageResponse, summary="Send a magic link")
async def request_magic_link(body: MagicLinkRequest) -> MessageResponse:
    """Email a one-time login link to the given address."""
    await send_magic_link(body.email)
    return MessageResponse(message="Magic link sent")


@router.get("/magic/verify", summary="Verify a magic link token")
async def verify_magic(token: str):
    email = verify_magic_link(token)
    if not email:
        raise InvalidCredentialError("magic link")

    response = RedirectResponse(url="/")
    await _authenticate_user(email=email, response=response, auth_provider="magic")
    return response


# --- Passwordless: OTP ---

@router.post("/otp/send", response_model=MessageResponse, summary="Send a one-time passcode")
async def request_otp(body: OTPSendRequest) -> MessageResponse:
    """Email a 6-digit verification code to the given address."""
    await send_otp(body.email)
    return MessageResponse(message="Verification code sent")


@router.post("/otp/verify", response_model=AuthResponse, summary="Verify a one-time passcode")
async def verify_otp_endpoint(body: OTPVerifyRequest, response: Response) -> AuthResponse:
    valid = await verify_otp(body.email, body.code)
    if not valid:
        raise InvalidCredentialError("verification code")

    result = await _authenticate_user(email=body.email, response=response, auth_provider="otp")
    return AuthResponse(message="Logged in successfully", user_id=result["user_id"])