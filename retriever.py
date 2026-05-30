import json
import numpy as np
from fastembed import TextEmbedding
from deep_translator import GoogleTranslator

INDIAN_STATES = [
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "jammu", "kashmir", "ladakh", "puducherry", "chandigarh",
    "andaman", "nicobar", "lakshadweep", "dadra", "daman", "diu"
]

# common abbreviations
STATE_ALIASES = {
    "up": "uttar pradesh",
    "mp": "madhya pradesh",
    "ap": "andhra pradesh",
    "tn": "tamil nadu",
    "wb": "west bengal",
    "hp": "himachal pradesh",
    "uk": "uttarakhand",
    "jk": "jammu and kashmir",
}

def translate_to_user_language(text, target_lang):
    try:
        if target_lang == 'en':
            return text  # no translation needed
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text  # fallback to english

# Maps lesser-used or script-variant languages to the practical language used for replies.
# Key insight: langdetect sees script, not intent — Urdu and Hindi are the same spoken
# language but different scripts, so whisper-transcribed Hindi often comes back as "ur".
LANGUAGE_MAP = {
    "ur": "hi",   # Urdu → Hindi  (same spoken language, Nastaliq vs Devanagari script)
    "ne": "hi",   # Nepali → Hindi (mutually intelligible, Devanagari)
    "sa": "hi",   # Sanskrit → Hindi
    "sd": "hi",   # Sindhi → Hindi (common in Rajasthan/Gujarat belt)
    "mai": "hi",  # Maithili → Hindi
    "as": "bn",   # Assamese → Bengali (near-identical script and grammar)
    "kok": "mr",  # Konkani → Marathi (dominant in Goa/Konkan coast)
    "si": "ta",   # Sinhala → Tamil  (Tamil is the practical fallback for South users)
    "pa": "hi",   # Punjabi → Hindi (most Punjabi speakers in India also understand Hindi)
}

def detect_language(text):
    try:
        from langdetect import detect
        lang = detect(text)
        lang = LANGUAGE_MAP.get(lang, lang)
        print(f"Detected language: {lang}")
        return lang
    except:
        return 'en'  # fallback to english

def translate_to_english(text):
    try:
        detected = GoogleTranslator(source='auto', target='en').translate(text)
        print(f"Translated: '{text}' → '{detected}'")
        return detected
    except:
        return text  # fallback to original if translation fails

def detect_state(query):
    q = query.lower().strip()
    # check aliases first
    for alias, state in STATE_ALIASES.items():
        if f" {alias} " in f" {q} ":
            return state
    # check full names
    for state in INDIAN_STATES:
        if state in q:
            return state
    return None

# add at top of file, outside class
QUERY_EXPANSIONS = {
    "lost my job":              "unemployment retrenchment job loss laid off worker employment benefit",
    "no job":                   "unemployment retrenchment job loss laid off worker employment benefit",
    "lost job":                 "unemployment retrenchment job loss laid off worker employment benefit",
    "need money":               "financial assistance poverty below poverty line BPL economic help",
    "no money":                 "financial assistance poverty below poverty line BPL economic help",
    "poor":                     "below poverty line BPL financial assistance low income",
    "crop damage":              "farmer crop loss flood drought agricultural compensation relief",
    "crop damaged":             "farmer crop loss flood drought agricultural compensation relief",
    "flood damage":             "flood relief compensation disaster natural calamity farmer",
    "pregnant":                 "maternity benefit pregnancy financial assistance mother child",
    "old age":                  "senior citizen elderly old age pension assistance",
    "disability":               "disabled person handicap differently abled assistance scheme",
    "student scholarship":      "scholarship stipend education financial assistance student",
    "business loan":            "entrepreneur MSME small business loan financial assistance",
}

def expand_query(query):
    q = query.lower().strip()
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in q:
            return expansion
    return query  # no expansion found, use original

class SchemeRetriever:
    def __init__(self, npz_path, jsonl_path="retrieval_documents.jsonl"):
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.scheme_ids = data["scheme_ids"].tolist()
        self.model      = TextEmbedding("BAAI/bge-small-en-v1.5")

        self.records = {}
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                self.records[r["scheme_id"]] = r

        print(f"Ready. {len(self.scheme_ids)} schemes loaded.")
        print(f"Records loaded: {len(self.records)}")

    def search(self, query, top_k=5, threshold=0.30, state_filter=None):
        query    = translate_to_english(query)
        expanded = expand_query(query)
        detected_state = detect_state(query)
        
        # use explicit filter or detected state
        active_state = state_filter or detected_state

        vec    = np.array(list(self.model.embed([expanded]))[0], dtype="float32")
        vec    = vec / np.linalg.norm(vec)
        scores = self.embeddings @ vec
        idx    = np.argsort(scores)[::-1]

        results = []
        for i in idx:
            sid   = self.scheme_ids[i]
            rec   = self.records.get(sid, {})
            score = float(scores[i])
            if score < threshold:
                break

            # state filter — keep central schemes + matching state
            scheme_state = rec.get("state", "").lower()
            if active_state:
                if scheme_state and active_state not in scheme_state:
                    print("skipping non matching state query")
                    continue  # skip non-matching state schemes

            results.append({
                "scheme_id":   sid,
                "scheme_name": rec.get("scheme_name", ""),
                "slug":        rec.get("slug", ""),
                "state":       rec.get("state", ""),
                "categories":  rec.get("categories", []),
                "tags":        rec.get("tags", []),
                "score":       score,
                "detected_state": active_state,
            })
            if len(results) == top_k:
                break
        return results