import json
import os
from datetime import datetime, timezone

LOGS_DIR    = "logs"
AUDIO_DIR   = os.path.join(LOGS_DIR, "audio")
LOG_CHAT_ID = os.environ.get("LOG_CHAT_ID", "")

os.makedirs(AUDIO_DIR, exist_ok=True)

def make_session_id(user_id: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{user_id}"

def audio_path(session_id: str) -> str:
    return os.path.join(AUDIO_DIR, f"{session_id}.ogg")

def write(entry: dict):
    date  = datetime.now(timezone.utc).strftime("%Y%m%d")
    fpath = os.path.join(LOGS_DIR, f"sessions_{date}.jsonl")
    try:
        with open(fpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # ephemeral filesystem on Railway — Telegram log is the source of truth

def format_entry(entry: dict) -> str:
    sid     = entry.get("session_id", "?")
    uid     = entry.get("user_id", "?")
    results = entry.get("results", [])

    if entry.get("type") == "text":
        lines = [
            f"📝 TEXT | {sid}",
            f"👤 {uid}",
            f"🔍 {entry.get('query', '')}",
        ]
    else:
        confirmed = "✅ Yes" if entry.get("user_confirmed") else "❌ Cancelled"
        lines = [
            f"🎙 VOICE | {sid}",
            f"👤 {uid}",
            f"🔊 whisper={entry.get('whisper_lang')} → {entry.get('mapped_lang')}",
            f"📝 raw: {entry.get('raw_transcript', '')}",
            f"🔤 query: {entry.get('final_query', '')}",
            f"🙋 confirmed: {confirmed}",
        ]

    if results:
        lines.append(f"✅ {len(results)} result(s):")
        for r in results:
            lines.append(f"  • {r['name']} ({r['score']:.2f}) — {r['state'] or 'All India'}")
    else:
        lines.append("❌ No results")

    return "\n".join(lines)
