import resend

from config import Config
from logging_config import get_logger

_log = get_logger("auth.email")


async def send_email(to: str, subject: str, html: str) -> None:
    if not Config.RESEND_API_KEY:
        _log.warning("email.skipped_no_api_key", to=to, subject=subject, html=html)
        return

    resend.api_key = Config.RESEND_API_KEY
    resend.Emails.send({
        "from": "noreply@resend.dev",
        "to": [to],
        "subject": subject,
        "html": html,
    })