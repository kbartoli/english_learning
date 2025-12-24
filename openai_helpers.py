import os
import json
import re
from typing import Any, Dict, Optional, List

import streamlit as st
from openai import OpenAI


# -----------------------------
# Basics / utilities
# -----------------------------
def check_api_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return (
            False,
            "Brakuje zmiennej środowiskowej OPENAI_API_KEY. "
            "Ustaw ją w systemie lub w pliku .env (lokalnie).",
        )
    return True, "OK"


def _client() -> OpenAI:
    # Official OpenAI Python SDK (new style)
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def safe_json_loads(text: str) -> Any:
    """
    Best-effort JSON parsing:
    - trims code fences
    - extracts first {...} or [...] block if model includes extra text
    """
    if not text:
        raise ValueError("Empty response; cannot parse JSON.")

    cleaned = text.strip()

    # Remove ```json fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # If there's extra text, try to extract the first JSON object/array
    # This is heuristic; structured prompting should make it unnecessary, but helps robustness.
    first_obj = None
    for pattern in [r"(\{.*\})", r"(\[.*\])"]:
        m = re.search(pattern, cleaned, flags=re.DOTALL)
        if m:
            first_obj = m.group(1)
            break
    candidate = first_obj if first_obj else cleaned

    return json.loads(candidate)


# -----------------------------
# Sentence generation
# -----------------------------
_SENTENCE_SYSTEM = """You are an expert English teacher for adult Polish speakers.
You must respond with STRICT JSON only. No markdown, no extra text.
All explanations must be in Polish, friendly and concise.
Content must be safe, non-hateful, non-sexual, non-violent, suitable for general audiences.
"""

_SENTENCE_USER_TEMPLATE = """Generate {n} short conversational English sentences related to this topic:

TOPIC: {topic}

Guidelines:
- Keep sentences natural for everyday conversation and adult learners.
- Prefer B1-ish by default unless target_level is specified.
- Avoid sensitive topics; keep it safe and practical.
- Return JSON with this exact schema:

{{
  "topic": "<string>",
  "items": [
    {{
      "sentence_en": "<string>",
      "explanation_pl": {{
        "translation_pl": "<string>",
        "key_vocab_pl": ["<string>", "..."],
        "grammar_note_pl": "<string>",
        "variation_en": "<string>"
      }},
      "difficulty_tag": "<A1|A2|B1|B2|C1>"
    }}
  ]
}}

If target_level is not "auto", match it.
Target level: {target_level}
"""


@st.cache_data(show_spinner=False, ttl=60 * 60)
def generate_sentences_json(topic: str, n: int = 5, target_level: str = "auto") -> Dict[str, Any]:
    if not topic.strip():
        raise ValueError("Topic is empty.")

    client = _client()
    prompt = _SENTENCE_USER_TEMPLATE.format(topic=topic.strip(), n=int(n), target_level=target_level)

    # Using Responses API (new style)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": _SENTENCE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    text = resp.output_text
    data = safe_json_loads(text)

    # Minimal validation
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError("Invalid JSON schema: missing 'items' list.")

    return data


# -----------------------------
# TTS (mp3)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def tts_mp3_bytes(text: str) -> bytes:
    if not text or not text.strip():
        raise ValueError("Empty text for TTS.")

    client = _client()
    # Official TTS endpoint (new SDK style): audio.speech.create
    # Note: voices may change; "alloy" is commonly available.
    audio = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        response_format="mp3",
        input=text.strip(),
    )
    return audio.read()


# -----------------------------
# Speech-to-text (transcription)
# -----------------------------
def transcribe_audio_to_text(file_path: str) -> str:
    """
    Uses OpenAI STT to transcribe the provided audio file.
    Note: Confidence/word-level timestamps are not always available depending on model/endpoint.
    """
    client = _client()
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    # SDK returns an object with .text in many cases
    text = getattr(transcript, "text", None)
    if not text:
        # Fallback: try dict-like
        try:
            text = transcript["text"]
        except Exception:
            pass
    return (text or "").strip()


# -----------------------------
# Analysis / feedback
# -----------------------------
_ANALYSIS_SYSTEM = """You are an expert English pronunciation and grammar coach for adult Polish learners.
You must respond with STRICT JSON only (no markdown, no extra text).
Write explanations in Polish, but keep corrected_version_en and transcript in English.
You CANNOT provide a numeric pronunciation score because true scoring requires specialized phonetic analysis.
You can provide heuristic pronunciation tips based on typical PL accent issues and the transcript.
Be kind, specific, and actionable.
"""

_ANALYSIS_USER_TEMPLATE = """We are practicing spoken English.

TOPIC: {topic}
REFERENCE SENTENCE (what the learner practiced / responded to): {prompt_sentence_en}

LEARNER TRANSCRIPT (from speech-to-text): {transcript}

Return JSON with EXACT schema:
{{
  "transcript": "<string>",
  "corrected_version_en": "<string>",
  "grammar_notes_pl": "<string>",
  "pronunciation_tips_pl": ["<bullet>", "..."],
  "next_steps_pl": ["<bullet>", "..."],
  "model_answer_en": "<string>",
  "model_answer_pl": "<string>"
}}

Requirements:
- corrected_version_en: rewrite learner meaning in natural English; keep it short and conversational.
- grammar_notes_pl: explain the main grammar fixes in Polish with 1-2 examples.
- pronunciation_tips_pl: give 4-7 tips, tie them to words in transcript when possible. Mention typical pitfalls:
  - -ed endings, final consonants, /θ/ /ð/ ("th"), vowel length (ship/sheep), word stress, linking, intonation.
  - If transcript suggests missing endings or articles, mention clarity tips.
- next_steps_pl: 3-5 concrete practice actions.
- model_answer: provide a strong example answer aligned to the topic/reference sentence, with Polish explanation.
- If the transcript is empty/very short, gently ask to re-record and give general tips.
"""


@st.cache_data(show_spinner=False, ttl=30 * 60)
def analyze_spoken_answer_json(topic: str, prompt_sentence_en: str, transcript: str) -> Dict[str, Any]:
    client = _client()

    t = (transcript or "").strip()
    if not t:
        # Still return a structured response without calling the model
        return {
            "transcript": "",
            "corrected_version_en": "",
            "grammar_notes_pl": "Nie udało się odczytać transkryptu. Spróbuj nagrać ponownie, mówiąc wolniej i wyraźniej.",
            "pronunciation_tips_pl": [
                "Nagraj w cichym miejscu, trzymaj mikrofon blisko.",
                "Mów wolniej i rób krótkie pauzy między frazami.",
                "Wyraźnie domykaj końcówki wyrazów (np. -s, -t, -d).",
                "Zadbaj o intonację w zdaniach pytających (wznosząca na końcu).",
            ],
            "next_steps_pl": [
                "Nagraj ponownie 1–2 zdania.",
                "Porównaj z nagraniem TTS i powtórz metodą shadowing (jednocześnie/tuż po).",
                "Skup się na końcówkach i rytmie zdania.",
            ],
            "model_answer_en": "Could you repeat your answer once more? I couldn't catch it clearly.",
            "model_answer_pl": "Prośba o powtórzenie – uprzejma i naturalna w rozmowie.",
        }

    prompt = _ANALYSIS_USER_TEMPLATE.format(
        topic=topic.strip(),
        prompt_sentence_en=(prompt_sentence_en or "").strip(),
        transcript=t,
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": _ANALYSIS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    data = safe_json_loads(resp.output_text)

    # Minimal validation
    for key in ["transcript", "corrected_version_en", "grammar_notes_pl", "pronunciation_tips_pl", "next_steps_pl"]:
        if key not in data:
            raise ValueError(f"Invalid JSON schema: missing '{key}'.")

    # Ensure lists are lists
    if not isinstance(data.get("pronunciation_tips_pl"), list):
        data["pronunciation_tips_pl"] = [str(data.get("pronunciation_tips_pl"))]
    if not isinstance(data.get("next_steps_pl"), list):
        data["next_steps_pl"] = [str(data.get("next_steps_pl"))]

    return data
