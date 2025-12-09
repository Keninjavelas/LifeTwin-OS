"""SQLAlchemy-backed storage for encrypted summaries (optional).

This module uses SQLAlchemy if available. If SQLAlchemy is not installed the
main app will continue to use the simple sqlite3-based `storage.py`.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List

try:
    from sqlalchemy import (
        create_engine,
        Column,
        Integer,
        String,
        Text,
        DateTime,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
except Exception:
    raise

Base = declarative_base()


class EncryptedSummary(Base):
    __tablename__ = "encrypted_summaries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String, index=True)
    created_at = Column(DateTime)
    algorithm = Column(String)
    wrapped_dek = Column(Text)
    ciphertext = Column(Text)
    metadata = Column(Text)


DB_URL = os.environ.get("LIFETWIN_ENC_DB_URL") or f"sqlite:///{os.path.join(os.path.dirname(__file__), '..', 'data', 'encrypted_store_sqlalchemy.sqlite')}"

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def store_encrypted_summary(device_id: str, algorithm: str, wrapped_dek_hex: str, ciphertext_hex: str, metadata: Optional[Dict[str, Any]] = None) -> int:
    init_db()
    session = SessionLocal()
    item = EncryptedSummary(
        device_id=device_id,
        created_at=datetime.utcnow(),
        algorithm=algorithm,
        wrapped_dek=wrapped_dek_hex,
        ciphertext=ciphertext_hex,
        metadata=json.dumps(metadata or {}),
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    rowid = item.id
    session.close()
    return rowid


def list_encrypted_summaries(limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    session = SessionLocal()
    rows = session.query(EncryptedSummary).order_by(EncryptedSummary.id.desc()).limit(limit).all()
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "device_id": r.device_id,
            "created_at": r.created_at.isoformat() + "Z",
            "algorithm": r.algorithm,
            "wrapped_dek": r.wrapped_dek,
            "ciphertext": r.ciphertext,
            "metadata": json.loads(r.metadata or "{}"),
        })
    session.close()
    return out


if __name__ == "__main__":
    print("Initializing SQLAlchemy encrypted store at:", DB_URL)
    init_db()
