import os
import json
import tempfile
from typing import Optional, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

from openai_helpers import (
    check_api_key,
    generate_sentences_json,
    tts_mp3_bytes,
    transcribe_audio_to_text,
    analyze_spoken_answer_json,
    safe_json_loads,
)

# Load .env for local dev (won't override real env vars)
load_dotenv()

st.set_page_config(page_title="English Speaking Practice (PL)", page_icon="🗣️", layout="wide")

st.title("🗣️ English Speaking Practice (z wyjaśnieniami po polsku)")
st.write(
    """
Wpisz temat, wygeneruj krótkie zdania konwersacyjne po angielsku + wyjaśnienia po polsku.
Odsłuchaj nagranie (TTS), a potem nagraj swoją odpowiedź i otrzymaj informację zwrotną
(dot. gramatyki oraz **przybliżone** wskazówki wymowy na podstawie transkrypcji).
"""
)

with st.expander("⚠️ Ograniczenia prototypu (ważne)", expanded=False):
    st.markdown(
        """
- Ocena wymowy jest **heurystyczna**: opiera się głównie na transkrypcji (speech-to-text) i typowych problemach PL->EN.
- Bez specjalistycznych usług fonetycznych nie da się rzetelnie policzyć „wyniku” wymowy.
- Najlepiej działa z wyraźną mową i w cichym pomieszczeniu.
"""
    )

api_ok, api_msg = check_api_key()
if not api_ok:
    st.error(api_msg)
    st.stop()

# --- Session state ---
if "generated" not in st.session_state:
    st.session_state.generated = None  # Dict[str, Any]
if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""
if "selected_sentence_en" not in st.session_state:
    st.session_state.selected_sentence_en = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None

# --- Topic input ---
col_left, col_right = st.columns([2, 1], gap="large")
with col_left:
    topic = st.text_area(
        "Temat (dowolny):",
        placeholder="Np. podróże, rozmowa kwalifikacyjna, zdrowe jedzenie, technologia w pracy...",
        height=90,
    )
with col_right:
    num_sentences = st.slider("Ile zdań wygenerować?", min_value=3, max_value=8, value=5, step=1)
    target_level = st.selectbox("Poziom (opcjonalnie):", ["auto", "A1", "A2", "B1", "B2", "C1"], index=0)

st.divider()

# --- Generate button ---
gen_col1, gen_col2 = st.columns([1, 3])
with gen_col1:
    generate_clicked = st.button("✨ Generate", use_container_width=True)
with gen_col2:
    st.caption("Wskazówka: wpisz konkretną sytuację, np. „small talk at a conference” – zdania będą lepiej dopasowane.")

if generate_clicked:
    if not topic.strip():
        st.warning("Wpisz temat, zanim klikniesz Generate.")
    else:
        with st.spinner("Generuję zdania i wyjaśnienia..."):
            try:
                result = generate_sentences_json(topic=topic.strip(), n=num_sentences, target_level=target_level)
                st.session_state.generated = result
                st.session_state.last_topic = topic.strip()
                st.session_state.analysis = None
                st.session_state.selected_sentence_en = None
            except Exception as e:
                st.error(f"Nie udało się wygenerować treści. Szczegóły: {e}")

generated = st.session_state.generated

# --- Display generated sentences ---
if generated:
    st.subheader("✅ Wygenerowane zdania")
    sentences: List[Dict[str, Any]] = generated.get("items", [])
    if not sentences:
        st.warning("Brak wyników. Spróbuj ponownie z innym tematem.")
    else:
        # Choose a sentence for speaking practice
        sentence_options = [s.get("sentence_en", "") for s in sentences if s.get("sentence_en")]
        st.session_state.selected_sentence_en = st.selectbox(
            "Wybierz zdanie do przećwiczenia (do modelowej odpowiedzi / analizy):",
            options=sentence_options,
            index=0 if sentence_options else None,
        )

        for idx, item in enumerate(sentences, start=1):
            sentence_en = item.get("sentence_en", "").strip()
            explanation_pl = item.get("explanation_pl", {})
            difficulty = item.get("difficulty_tag", "auto")

            title = f"{idx}. [{difficulty}] {sentence_en}" if sentence_en else f"{idx}. (brak zdania)"
            with st.expander(title, expanded=(idx == 1)):
                if sentence_en:
                    st.markdown(f"**EN:** {sentence_en}")
                else:
                    st.warning("Brak sentence_en w danych.")

                # Explanation can be dict (preferred) but may come as string if model deviates
                if isinstance(explanation_pl, dict):
                    translation = explanation_pl.get("translation_pl", "")
                    vocab = explanation_pl.get("key_vocab_pl", [])
                    grammar = explanation_pl.get("grammar_note_pl", "")
                    variation = explanation_pl.get("variation_en", "")

                    if translation:
                        st.markdown(f"**PL (tłumaczenie):** {translation}")
                    if vocab:
                        st.markdown("**Słownictwo (kluczowe):**")
                        for v in vocab:
                            # expects strings like "word — meaning"
                            st.markdown(f"- {v}")
                    if grammar:
                        st.markdown(f"**Gramatyka (PL):** {grammar}")
                    if variation:
                        st.markdown(f"**Wariant (EN):** {variation}")
                else:
                    st.markdown("**Wyjaśnienie (PL):**")
                    st.write(explanation_pl)

                # TTS audio (cached by helper)
                try:
                    audio_bytes = tts_mp3_bytes(sentence_en)
                    st.audio(audio_bytes, format="audio/mp3")
                    st.caption("Audio generowane przez OpenAI TTS (mp3).")
                except Exception as e:
                    st.error(f"Nie udało się wygenerować audio (TTS): {e}")

st.divider()

# --- Recording section ---
st.subheader("🎙️ Nagraj swoją odpowiedź")

st.write(
    "Nagraj audio (WAV/MP3/M4A). Jeśli nie masz wbudowanego nagrywania w Streamlit, skorzystaj z **Upload audio**."
)

use_mic_plugin = st.toggle(
    "Użyj streamlit-mic-recorder (jeśli zainstalowane)",
    value=True,
    help="Jeśli w Twoim środowisku nie działa, wyłącz i użyj uploadu pliku.",
)

audio_bytes: Optional[bytes] = None
audio_mime: Optional[str] = None

if use_mic_plugin:
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore

        mic = mic_recorder(
            start_prompt="⏺️ Start recording",
            stop_prompt="⏹️ Stop",
            just_once=False,
            use_container_width=True,
            callback=None,
            format="wav",
        )
        if mic and isinstance(mic, dict) and mic.get("bytes"):
            audio_bytes = mic["bytes"]
            audio_mime = mic.get("mime_type") or "audio/wav"
            st.success("Nagranie zostało przechwycone.")
            st.audio(audio_bytes, format=audio_mime)
    except Exception:
        st.info(
            "Nie wykryto lub nie udało się użyć streamlit-mic-recorder. "
            "Wyłącz przełącznik powyżej i użyj uploadu pliku."
        )

if audio_bytes is None:
    uploaded = st.file_uploader("Upload audio:", type=["wav", "mp3", "m4a", "mp4", "webm", "ogg"])
    if uploaded is not None:
        audio_bytes = uploaded.read()
        audio_mime = uploaded.type or "audio/*"
        st.audio(audio_bytes, format=audio_mime)

analyze_clicked = st.button("🧠 Analyze my answer", use_container_width=True)

if analyze_clicked:
    if not generated:
        st.warning("Najpierw wygeneruj zdania (Generate).")
    elif not audio_bytes:
        st.warning("Dodaj nagranie audio przed analizą.")
    else:
        # Save bytes to temp file (some STT flows need a file-like object with name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        prompt_sentence = st.session_state.selected_sentence_en or ""
        topic_for_analysis = st.session_state.last_topic or topic.strip()

        with st.spinner("Transkrybuję i analizuję odpowiedź..."):
            try:
                transcript = transcribe_audio_to_text(tmp_path)
                analysis = analyze_spoken_answer_json(
                    topic=topic_for_analysis,
                    prompt_sentence_en=prompt_sentence,
                    transcript=transcript,
                )
                st.session_state.analysis = analysis
            except Exception as e:
                st.error(f"Nie udało się przeanalizować odpowiedzi: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

analysis = st.session_state.analysis
if analysis:
    st.subheader("📌 Feedback")
    st.markdown("**Transkrypt (EN):**")
    st.write(analysis.get("transcript", ""))

    st.markdown("**Poprawiona wersja (EN):**")
    st.write(analysis.get("corrected_version_en", ""))

    st.markdown("**Uwagi gramatyczne (PL):**")
    st.write(analysis.get("grammar_notes_pl", ""))

    tips = analysis.get("pronunciation_tips_pl", [])
    st.markdown("**Wskazówki wymowy (PL) – heurystyczne:**")
    if isinstance(tips, list) and tips:
        for t in tips:
            st.markdown(f"- {t}")
    else:
        st.write(tips)

    next_steps = analysis.get("next_steps_pl", [])
    st.markdown("**Następne kroki (PL):**")
    if isinstance(next_steps, list) and next_steps:
        for n in next_steps:
            st.markdown(f"- {n}")
    else:
        st.write(next_steps)

    # Optional: model answer
    model_answer_en = analysis.get("model_answer_en", "")
    model_answer_pl = analysis.get("model_answer_pl", "")
    if model_answer_en or model_answer_pl:
        st.markdown("**Model answer (EN):**")
        st.write(model_answer_en)
        st.markdown("**Wyjaśnienie modelowej odpowiedzi (PL):**")
        st.write(model_answer_pl)

st.divider()
st.caption("Prototyp lokalny • Streamlit + OpenAI SDK • Sekrety przez OPENAI_API_KEY")
