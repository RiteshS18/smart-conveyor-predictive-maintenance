"""SQLite persistence (SQLAlchemy 2.0)."""
from __future__ import annotations

import contextlib
from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATA_DIR

engine = create_engine(
    f"sqlite:///{DATA_DIR / 'conveyor.db'}",
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_con, _rec):  # WAL = readers don't block the writer
    cur = dbapi_con.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[float] = mapped_column(Float, index=True)       # unix seconds
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vibration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    load: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tension: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="http")


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    severity: Mapped[str] = mapped_column(String(10))           # info | warning | critical
    kind: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(String(300))
    snapshot: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class FeatureRow(Base):
    """One row every ~5 s: the fused vision + sensor state used for prediction."""
    __tablename__ = "feature_rows"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[float] = mapped_column(Float, index=True)
    n_defects: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_conf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    misalign_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dust_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    blur: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vibration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    load: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tension: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    health: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anomaly: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)


Base.metadata.create_all(engine)


@contextlib.contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
