from pydantic import EmailStr, Field

from models.meta import MongoDocument


class User(MongoDocument):
    email: EmailStr
    first_name: str = Field(min_length=1)
    middle_names: list[str] = []
    last_name: str = Field(min_length=1)
    auth_provider: str | None = None
    email_verified: bool = False

    @property
    def full_name(self) -> str:
        parts = [self.first_name, *self.middle_names, self.last_name]
        return " ".join(parts)
