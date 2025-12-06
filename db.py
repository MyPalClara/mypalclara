from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base

# SQLite by default; swap out URL for Postgres if you want.
DATABASE_URL = "sqlite:///assistant.db"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
