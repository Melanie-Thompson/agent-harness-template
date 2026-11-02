from unittest.mock import MagicMock, patch

import pytest

from auth.email import send_email


async def test_send_email_calls_resend():
    with (
        patch("auth.email.Config") as mock_config,
        patch("auth.email.resend") as mock_resend,
    ):
        mock_config.RESEND_API_KEY = "test-key"
        mock_resend.Emails = MagicMock()

        await send_email("user@example.com", "Test Subject", "<p>Hello</p>")

        mock_resend.Emails.send.assert_called_once_with({
            "from": "noreply@resend.dev",
            "to": ["user@example.com"],
            "subject": "Test Subject",
            "html": "<p>Hello</p>",
        })


async def test_send_email_sets_api_key():
    with (
        patch("auth.email.Config") as mock_config,
        patch("auth.email.resend") as mock_resend,
    ):
        mock_config.RESEND_API_KEY = "test-api-key"
        mock_resend.Emails = MagicMock()

        await send_email("user@example.com", "Subject", "<p>Hi</p>")

        assert mock_resend.api_key == "test-api-key"


async def test_send_email_propagates_errors():
    with (
        patch("auth.email.Config") as mock_config,
        patch("auth.email.resend") as mock_resend,
    ):
        mock_config.RESEND_API_KEY = "test-key"
        mock_resend.Emails.send.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            await send_email("user@example.com", "Subject", "<p>Hi</p>")


async def test_send_email_skips_when_no_api_key():
    with (
        patch("auth.email.Config") as mock_config,
        patch("auth.email.resend") as mock_resend,
    ):
        mock_config.RESEND_API_KEY = ""
        mock_resend.Emails = MagicMock()

        await send_email("user@example.com", "Subject", "<p>Hi</p>")

        mock_resend.Emails.send.assert_not_called()
