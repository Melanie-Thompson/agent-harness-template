# Basic Harness

A starting-point FastAPI template for building more elaborate harnesses. Production-ready auth, audited MongoDB, Redis sessions, and permission-based access control — ready to extend.

## Features

- **Auth** — OAuth (Google + GitHub), magic link, email OTP. No passwords.
- **Sessions** — Secure server-side sessions in Redis with httponly cookies
- **Permissions** — Fine-grained RBAC. Each endpoint declares required permissions via `Depends(require("read:reports"))`
- **MongoDB** — Async via Motor. Every document carries audited `_meta` (created/updated/deleted timestamps, user tracking, change deltas)
- **User scoping** — All queries automatically filtered to the logged-in user
- **Soft deletes** — Deletes set `_meta.deleted_at`, excluded from queries automatically
- **Pydantic everywhere** — Validated request/response schemas, domain models, and Meta audit trail
- **Auto-discovered routes** — Drop a file in `routes/` with a `router`, it's registered automatically

## Project Structure

```
main.py                  # FastAPI app, lifespan, middleware
config.py                # Env-backed Config class
auth/
  oauth.py               # Google + GitHub OAuth via Authlib
  passwordless.py         # Magic link + OTP logic
  email.py               # Email sending abstraction (Resend)
  permissions.py          # require() dependency for RBAC
db/
  connect.py             # DbConnect class + universal query actualiser
  meta.py                # Meta builder functions (create/update/delete)
  mongo.py               # Motor client lifecycle
  objectid.py            # ObjectId serialisation (single source of truth)
  sessions.py            # Redis session CRUD
models/
  meta.py                # Pydantic models: Change, Meta, MongoDocument
  user.py                # User model
schemas/
  auth.py                # Auth request/response models
  db.py                  # QueryInput, QueryResult, Operation enum
  session.py             # SessionData, MessageResponse, MeResponse, HealthResponse
routes/
  __init__.py            # Auto-discovers all routers in this package
  auth.py                # Auth endpoints (OAuth, magic link, OTP)
  sessions.py            # Logout, /me
tests/                   # 32 tests across 6 files
```

## Setup

### Prerequisites

- Python 3.13+
- Redis
- MongoDB

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials. For OAuth, set `GOOGLE_CLIENT_ID`/`SECRET` and `GITHUB_CLIENT_ID`/`SECRET`. For email, set `RESEND_API_KEY`. Generate a secret for magic links:

```bash
openssl rand -hex 32
```

### Run

```bash
uv run uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

### Test

```bash
uv sync --extra test
uv run pytest tests/ -v
```

## API Endpoints

### Health
- `GET /health` — `{"status": "ok"}`

### Auth
- `GET /auth/google` — Google OAuth login
- `GET /auth/github` — GitHub OAuth login
- `POST /auth/magic-link` — `{"email": "..."}` → sends magic link
- `POST /auth/otp/send` — `{"email": "..."}` → sends 6-digit code
- `POST /auth/otp/verify` — `{"email": "...", "code": "123456"}` → creates session

### Sessions
- `GET /sessions/me` — current user info (requires `read:self`)
- `POST /sessions/logout` — destroys session

## Adding New Endpoints

Create a file in `routes/` with a `router`:

```python
from fastapi import APIRouter, Depends
from auth import require
from db.connect import DbConnect, get_db

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")
async def list_items(db: DbConnect = Depends(get_db)):
    return await db.query(QueryInput(operation="find", collection="items"))
```

It's auto-registered — no changes to `main.py` needed.

## Adding New Models

Extend `MongoDocument` in `models/`:

```python
from models.meta import MongoDocument

class Order(MongoDocument):
    product: str
    quantity: int
    total: float
```

The `_meta` audit trail and `_id` are inherited automatically.
