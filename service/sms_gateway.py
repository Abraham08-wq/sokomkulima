"""
Sending SMS. With an Africa's Talking account the message really goes out; without one it is kept in the
OUTBOX (shown on the admin page), so the whole alert system can be demonstrated with no cost.

    AT_USERNAME=sandbox  AT_API_KEY=...            -> Africa's Talking SANDBOX (free; see their online simulator)
    AT_USERNAME=<your app> AT_API_KEY=... AT_FROM=  -> real SMS (paid), optional sender ID / short code
"""
import json, os, urllib.parse, urllib.request
from . import db

USER = os.environ.get("AT_USERNAME", "")
KEY = os.environ.get("AT_API_KEY", "")
FROM = os.environ.get("AT_FROM", "")


def configured():
    return bool(USER and KEY)


def _send_at(phone, message):
    host = "api.sandbox.africastalking.com" if USER == "sandbox" else "api.africastalking.com"
    body = {"username": USER, "to": phone, "message": message}
    if FROM:
        body["from"] = FROM
    req = urllib.request.Request(f"https://{host}/version1/messaging", data=urllib.parse.urlencode(body).encode(),
                                 headers={"apiKey": KEY, "Accept": "application/json",
                                          "Content-Type": "application/x-www-form-urlencoded"})
    r = json.loads(urllib.request.urlopen(req, timeout=30).read())
    rec = (r.get("SMSMessageData", {}).get("Recipients") or [{}])[0]
    return rec.get("status", "unknown"), json.dumps(r)[:300]


def send(farmer, kind, key, message):
    """Send once per (farmer, kind, key). Returns the stored message row, or None if already sent."""
    if db.q("SELECT id FROM messages WHERE farmer_id=? AND kind=? AND key=?", (farmer["id"], kind, key), one=True):
        return None
    status, detail = "outbox", "No SMS gateway configured: kept in the outbox"
    if configured() and farmer.get("phone"):
        try:
            status, detail = _send_at(farmer["phone"], message)
        except Exception as e:
            status, detail = "failed", f"{type(e).__name__}: {e}"[:300]
    mid = db.insert("INSERT OR IGNORE INTO messages(farmer_id, phone, kind, key, message, status, detail, created) "
                    "VALUES(?,?,?,?,?,?,?,?)", (farmer["id"], farmer.get("phone"), kind, key, message, status, detail, db.now()))
    return {"id": mid, "kind": kind, "message": message, "status": status}
