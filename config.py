import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    SESSION_EXPIRE_SECONDS: int = int(os.getenv("SESSION_EXPIRE_SECONDS", 3600 * 8))
    COOKIE_SECURE: bool = os.getenv("COOKIE_SECURE", "false").lower() == "true"
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    MONGO_DB: str = os.getenv("MONGO_DB", "tda_harness")
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
    MAGIC_LINK_SECRET: str = os.getenv("MAGIC_LINK_SECRET", "change-me-in-production")
    MAGIC_LINK_EXPIRE: int = int(os.getenv("MAGIC_LINK_EXPIRE", 600))
    OTP_EXPIRE: int = int(os.getenv("OTP_EXPIRE", 300))
    APP_URL: str = os.getenv("APP_URL", "http://localhost:8000")