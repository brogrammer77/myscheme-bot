import json
import os
from datetime import datetime, timezone

LOGS_DIR  = "logs"
AUDIO_DIR = os.path.join(LOGS_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

def make_session_id(user_id: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{user_id}"

def audio_path(session_id: str) -> str:
    return os.path.join(AUDIO_DIR, f"{session_id}.ogg")

def write(entry: dict):
    date  = datetime.now(timezone.utc).strftime("%Y%m%d")
    fpath = os.path.join(LOGS_DIR, f"sessions_{date}.jsonl")
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
