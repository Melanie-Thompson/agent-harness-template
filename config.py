import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    DRAGONFLY_URL: str = os.getenv("DRAGONFLY_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    CONVERSATION_STORE_URL: str = os.getenv(
        "HARNESS_CONVERSATION_STORE_URL",
        os.getenv("DRAGONFLY_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")),
    )
    CONVERSATION_TTL: int = int(os.getenv("HARNESS_CONVERSATION_TTL", "1800"))
    CONTEXT_MAX_MESSAGES: int = int(os.getenv("CONTEXT_MAX_MESSAGES", "50"))
    CONTEXT_MAX_TOKENS: int = int(os.getenv("CONTEXT_MAX_TOKENS", "4000"))
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
    GUARDRAILS_ENABLED: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
    GUARDRAILS_CONFIG_PATH: str = os.getenv("GUARDRAILS_CONFIG_PATH", "")
    GUARDRAILS_MODEL: str = os.getenv("GUARDRAILS_MODEL", "")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0))
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")