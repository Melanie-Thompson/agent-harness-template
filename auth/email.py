import resend

from config import Config


async def send_email(to: str, subject: str, html: str) -> None:
    resend.api_key = Config.RESEND_API_KEY
    resend.Emails.send({
        "from": "noreply@resend.dev",
        "to": [to],
        "subject": subject,
        "html": html,
    })