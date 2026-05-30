
# MyScheme Bot

Telegram bot for discovering Indian government welfare schemes. Describe your situation in plain language — text or voice — and the bot finds relevant schemes.

## Stack
- `faster-whisper` — offline voice transcription
- `fastembed` (BAAI/bge-small-en-v1.5) — semantic search over 2927 schemes
- `deep-translator` — multi-language support via Google Translate
- `python-telegram-bot` — Telegram interface

## Features
- **Text & voice search** — send a message or a voice note
- **Voice confirmation** — bot shows transcript before searching (✅/❌)
- **Multi-language** — detects Hindi, Bengali, Tamil, Gujarati, Marathi, etc.; replies in same language
- **Language normalisation** — maps Urdu/Punjabi/Nepali/Assamese to practical Indic languages
- **State-aware** — mention your state for local schemes; remembers it across queries
- **Session logging** — every interaction logged to `logs/sessions_YYYYMMDD.jsonl`; voice audio saved to `logs/audio/` for replay

## Run

```bash
pip install faster-whisper python-telegram-bot deep-translator fastembed langdetect numpy
PYTHONUNBUFFERED=1 python bot.py
```

## Files
| File | Purpose |
|---|---|
| `bot.py` | Telegram bot — text + voice handlers |
| `retriever.py` | Semantic search, translation, state detection |
| `logger.py` | Session logging (JSONL + audio files) |
| `scheme_embeddings.npz` | Precomputed embeddings (384-dim, L2-normalised) |
| `retrieval_documents.jsonl` | 2927 scheme records |

## Session Log Format
```json
{
  "session_id": "20260517_195423_987654321",
  "type": "voice",
  "whisper_lang": "hi",
  "raw_transcript": "...",
  "final_query": "मेरी फसल बाढ़ में तबाह हो गई",
  "user_confirmed": true,
  "results": [{"name": "...", "slug": "...", "score": 0.71, "state": "..."}]
}
```
