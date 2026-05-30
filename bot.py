import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import tempfile
import shutil
from datetime import datetime, timezone
from faster_whisper import WhisperModel
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from deep_translator import GoogleTranslator
from retriever import SchemeRetriever, detect_state, translate_to_english, translate_to_user_language, detect_language, LANGUAGE_MAP
import logger


TOKEN    = "8685315092:AAGJMx-9uDQj5F6G0HLatRRRXCM-g7ovPOY"
retriever = SchemeRetriever("scheme_embeddings.npz")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

user_state = {}
user_last_query = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Namaste! I help you find Indian government schemes you may be eligible for.\n\n"
        "Just tell me your situation in plain words. Examples:\n"
        "• lost my job\n"
        "• need help paying college fees\n"
        "• pregnant and need financial help\n"
        "• crop damaged due to flood\n\n"
        "You can also send a 🎙 voice message!\n\n"
        "What's your situation?"
    )

async def _search_and_reply(message, user_id: int, query: str, now: float):
    user_lang = detect_language(query)

    # rate limit — 1 query per 5 seconds
    last = user_last_query.get(user_id, 0)
    if now - last < 5:
        await message.reply_text("⏳ Please wait a moment before sending another query.")
        return []
    user_last_query[user_id] = now

    # too short
    if len(query) < 5:
        await message.reply_text(
            "Please describe your situation in a few words.\n"
            "Example: *lost my job* or *need help for college fees*\n\n"
            "Type /help to see more examples.",
            parse_mode="Markdown"
        )
        return []

    # too long
    if len(query) > 300:
        await message.reply_text(
            "Please keep your query short — one or two sentences is enough."
        )
        return []

    await message.reply_text("🔍 Searching schemes...")

    detected = detect_state(query)
    if detected:
        user_state[user_id] = detected

    active_state = detected or user_state.get(user_id)

    try:
        hits = retriever.search(query, top_k=3, state_filter=active_state)
    except Exception as e:
        await message.reply_text("⚠️ Something went wrong. Please try again in a moment.")
        print(f"Search error: {e}")
        return []

    if not hits:
        no_result_msg = "😔 No schemes found for your situation.\n\nTry using different words or mention your state."
        await message.reply_text(translate_to_user_language(no_result_msg, user_lang))
        return []

    if active_state:
        state_msg = f"📍 Showing schemes for {active_state.title()} + Central schemes"
        await message.reply_text(translate_to_user_language(state_msg, user_lang))

    await message.reply_text(f"✅ Found {len(hits)} relevant schemes:")

    for i, hit in enumerate(hits, 1):
        state = f"📍 {hit['state']}" if hit.get('state') else "📍 All India"
        cats  = ", ".join(hit['categories']) if hit.get('categories') else "General"

        scheme_name = translate_to_user_language(hit['scheme_name'], user_lang)
        cats        = translate_to_user_language(cats, user_lang)
        state       = translate_to_user_language(state, user_lang)

        msg = (
            f"*{i}. {scheme_name}*\n"
            f"{state}\n"
            f"🏷 {cats}\n\n"
            f"🔗 myscheme.gov.in/schemes/{hit['slug']}"
        )
        await message.reply_text(msg, parse_mode="Markdown")

    if not active_state:
        nudge = "💡 Mention your state for more relevant local schemes. Example: I am from Bihar"
        await message.reply_text(translate_to_user_language(nudge, user_lang))

    return hits

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id    = update.message.from_user.id
    query      = update.message.text.strip()
    now        = time.time()
    session_id = logger.make_session_id(user_id)
    timestamp  = datetime.now(timezone.utc).isoformat()

    print(f"[{session_id}] TEXT query: {query!r}")

    hits = await _search_and_reply(update.message, user_id, query, now)

    logger.write({
        "session_id":  session_id,
        "timestamp":   timestamp,
        "user_id":     user_id,
        "type":        "text",
        "query":       query,
        "results":     [{"name": h["scheme_name"], "slug": h["slug"],
                         "score": round(h["score"], 4), "state": h["state"]} for h in hits],
    })

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id    = update.message.from_user.id
    session_id = logger.make_session_id(user_id)
    timestamp  = datetime.now(timezone.utc).isoformat()

    print(f"[{session_id}] VOICE received")
    await update.message.reply_text("🎙 Processing your voice message...")

    voice_file  = await update.message.voice.get_file()
    saved_audio = logger.audio_path(session_id)

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await voice_file.download_to_drive(tmp_path)
        # keep a permanent copy for review
        shutil.copy2(tmp_path, saved_audio)

        segments, info = whisper_model.transcribe(tmp_path)
        raw_transcript  = " ".join(seg.text for seg in segments).strip()
        whisper_lang    = info.language
        mapped_lang     = LANGUAGE_MAP.get(whisper_lang, whisper_lang)

        print(f"[{session_id}] whisper_lang={whisper_lang} mapped_lang={mapped_lang} raw={raw_transcript!r}")

        # Always normalise the transcript to the mapped language so "I heard"
        # shows the correct script — whisper may detect 'hi' but write Urdu/Nastaliq.
        try:
            query = GoogleTranslator(source="auto", target=mapped_lang).translate(raw_transcript)
            print(f"[{session_id}] normalised to {mapped_lang}: {query!r}")
        except Exception:
            query = raw_transcript
    except Exception as e:
        await update.message.reply_text("⚠️ Could not process your voice message. Please try again.")
        print(f"[{session_id}] Whisper error: {e}")
        return
    finally:
        os.remove(tmp_path)

    if not query:
        await update.message.reply_text("⚠️ Could not understand your voice message. Please try again.")
        return

    # persist session data for the callback
    context.user_data["voice_session"] = {
        "session_id":     session_id,
        "timestamp":      timestamp,
        "audio_file":     saved_audio,
        "whisper_lang":   whisper_lang,
        "mapped_lang":    mapped_lang,
        "raw_transcript": raw_transcript,
        "final_query":    query,
    }
    context.user_data["pending_voice_query"] = query

    keyboard = [[
        InlineKeyboardButton("✅ Yes, search this", callback_data="voice_confirm"),
        InlineKeyboardButton("❌ Try again",        callback_data="voice_cancel"),
    ]]
    await update.message.reply_text(
        f"🎙 I heard:\n_{query}_\n\nIs this correct?",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )

async def handle_voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cb = update.callback_query
    await cb.answer()

    session = context.user_data.get("voice_session", {})
    query   = context.user_data.get("pending_voice_query")

    if cb.data == "voice_cancel":
        await cb.edit_message_text(
            "❌ No problem — please type your query as a text message or send a new voice note."
        )
        logger.write({**session, "user_confirmed": False, "results": []})
        print(f"[{session.get('session_id')}] user cancelled voice query")
        return

    if not query:
        await cb.edit_message_text("⚠️ Session expired. Please send your voice message again.")
        return

    await cb.edit_message_reply_markup(reply_markup=None)

    print(f"[{session.get('session_id')}] user confirmed, searching: {query!r}")
    hits = await _search_and_reply(cb.message, cb.from_user.id, query, time.time())

    logger.write({
        **session,
        "user_confirmed": True,
        "results": [{"name": h["scheme_name"], "slug": h["slug"],
                     "score": round(h["score"], 4), "state": h["state"]} for h in hits],
    })

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(CallbackQueryHandler(handle_voice_callback, pattern="^voice_(confirm|cancel)$"))
    print("Bot running...")
    app.run_polling()
