"""One-time, idempotent bootstrap for a *synthetic-demo-only* hosted instance.

Required secret variables are supplied through the Render dashboard, NEVER committed.
No account password is printed and an existing account is never reset on restart.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

from sqlalchemy import select
from app.core.config import check_secret
from app.database.db import Base, engine, SessionLocal
from app.database.models import User
from app.security.auth import hash_password


def main():
    check_secret()
    email = os.environ.get('BOOTSTRAP_ADMIN_EMAIL', '').strip().lower()
    password = os.environ.get('BOOTSTRAP_ADMIN_PASSWORD', '')
    if not email or '@' not in email or len(email) > 255:
        raise SystemExit('BOOTSTRAP_ADMIN_EMAIL must be set to a valid admin email in Render Environment.')
    if len(password) < 12:
        raise SystemExit('BOOTSTRAP_ADMIN_PASSWORD must be set to a secure 12+ character value in Render Environment.')
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        existing = db.scalar(select(User).where(User.email == email))
        if existing:
            if existing.role != 'admin':
                raise SystemExit('Bootstrap email is already used by a non-admin account. Choose another email.')
            print('Cloud bootstrap: existing administrator preserved (password unchanged).')
            return
        db.add(User(email=email, password_hash=hash_password(password), role='admin', active=True))
        db.commit()
    print('Cloud bootstrap: administrator created successfully. Keep credentials private.')


if __name__ == '__main__':
    main()
