"""
The farmer database (SQLite, data/sokomkulima.db). Farmers, cooperatives and every SMS sent are kept here.
SQLite needs no server; the file survives restarts because data/ is a Docker volume.
"""
from pathlib import Path
from datetime import datetime, timezone
import sqlite3, threading

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "sokomkulima.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS coops (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, crop TEXT NOT NULL, place TEXT, lat REAL, lon REAL,
    discount_pct REAL DEFAULT 10, created TEXT);
CREATE TABLE IF NOT EXISTS farmers (
    id INTEGER PRIMARY KEY, phone TEXT UNIQUE, name TEXT, lang TEXT DEFAULT 'en', crop TEXT NOT NULL,
    place TEXT, lat REAL, lon REAL, acres REAL DEFAULT 1, coop_id INTEGER REFERENCES coops(id),
    active INTEGER DEFAULT 1, created TEXT);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, farmer_id INTEGER, phone TEXT, kind TEXT, key TEXT, message TEXT,
    status TEXT, detail TEXT, created TEXT, UNIQUE(farmer_id, kind, key));
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def q(sql, args=(), one=False):
    with _lock:
        c = connect()
        try:
            cur = c.execute(sql, args); c.commit()
            rows = [dict(r) for r in cur.fetchall()]
            return (rows[0] if rows else None) if one else rows
        finally:
            c.close()


def insert(sql, args=()):
    with _lock:
        c = connect()
        try:
            cur = c.execute(sql, args); c.commit()
            return cur.lastrowid
        finally:
            c.close()


def norm_phone(p):
    """Keep digits and a leading +. Kenyan 07.. numbers become +2547.. (most gateways need the + form)."""
    p = "".join(ch for ch in str(p or "") if ch.isdigit() or ch == "+")
    if p.startswith("0") and len(p) == 10:
        p = "+254" + p[1:]
    if p and not p.startswith("+"):
        p = "+" + p
    return p
