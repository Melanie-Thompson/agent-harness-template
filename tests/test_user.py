from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models.user import User


def _meta(user_id="u1", by="system"):
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "created_at": now,
        "created_by": by,
        "updated_at": now,
        "updated_by": by,
    }


def test_user_basic():
    u = User(
        email="test@example.com",
        first_name="Jane",
        last_name="Doe",
        _meta=_meta(),
    )
    assert u.email == "test@example.com"
    assert u.first_name == "Jane"
    assert u.last_name == "Doe"
    assert u.middle_names == []
    assert u.auth_provider is None
    assert u.email_verified is False


def test_user_full_name_no_middle():
    u = User(email="a@b.com", first_name="Jane", last_name="Doe", _meta=_meta())
    assert u.full_name == "Jane Doe"


def test_user_full_name_with_middle():
    u = User(
        email="a@b.com",
        first_name="Jane",
        middle_names=["Marie", "Anne"],
        last_name="Doe",
        _meta=_meta(),
    )
    assert u.full_name == "Jane Marie Anne Doe"


def test_user_all_fields():
    u = User(
        email="a@b.com",
        first_name="Jane",
        middle_names=["M"],
        last_name="Doe",
        auth_provider="google",
        email_verified=True,
        _meta=_meta(),
    )
    assert u.auth_provider == "google"
    assert u.email_verified is True


def test_user_invalid_email():
    with pytest.raises(ValidationError):
        User(email="not-an-email", first_name="J", last_name="D", _meta=_meta())


def test_user_empty_first_name():
    with pytest.raises(ValidationError):
        User(email="a@b.com", first_name="", last_name="Doe", _meta=_meta())


def test_user_empty_last_name():
    with pytest.raises(ValidationError):
        User(email="a@b.com", first_name="Jane", last_name="", _meta=_meta())
