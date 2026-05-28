"""
app.py — PhoBuddy Streamlit Frontend
=====================================
Uses LangGraph agent (create_react_agent + MemorySaver).
Run with: streamlit run app.py
"""

import os
import uuid
import re
import hashlib
import streamlit as st
import requests as req
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from ingest import ingest_single_video
from agent import invoke_agent
from agent import build_agent

from dotenv import load_dotenv


load_dotenv()


# ── Helper functions ──────────────────────────────────────────────────────────

def get_config(name: str, default: str | None = None) -> str | None:
    """Read config from .env locally or Streamlit Secrets in deployment."""
    value = os.getenv(name)
    if value and not value.startswith("YOUR_"):
        return value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value and not str(secret_value).startswith("YOUR_"):
        return str(secret_value)
    return default


def clean_query(query: str) -> str:
    """Reframe photo/image requests as information requests."""
    patterns = [
        (r"can you show me (the )?(photo|image|picture) of (.+)", r"what is \3"),
        (r"show me (the )?(photo|image|picture) of (.+)", r"what is \3"),
        (r"(the )?(photo|image|picture) of (.+)", r"what is \3"),
        (r"^show me (.+)", r"what is \1"),
        (r"^can you show me (.+)", r"what is \1"),
    ]
    cleaned = query
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if result != cleaned:
            return result.strip()
    return query


def display_image(url: str) -> None:
    """Render images across both old and new Streamlit versions."""
    try:
        st.image(url, use_container_width=True)
    except TypeError:
        st.image(url, use_column_width=True)


PRONUNCIATION_HINTS = [
    (r"\bpho(\s+bo)?\b", "Pho Bo is pronounced fuh baw."),
    (r"\bbun bo hue\b", "Bun Bo Hue is pronounced boon baw hway."),
    (r"\bgoi cuon\b|\bspring roll", "Goi Cuon is pronounced goy koo-un."),
    (r"\bpad thai\b", "Pad Thai is pronounced pad tie."),
    (r"\btom yum(\s+goong)?\b", "Tom Yum Goong is pronounced tom yum goong."),
    (r"\bpad kra pao\b|\bbasil chicken\b", "Pad Kra Pao is pronounced pad kra pow."),
    (r"\bmapo tofu\b", "Mapo Tofu is pronounced mah-po tofu."),
    (r"\bchar siu\b", "Char Siu is pronounced char syoo."),
    (r"\bdumpling", "Dumplings are pronounced duhm-plings."),
    (r"\blemongrass\b", "Lemongrass is pronounced lemon grass."),
]


def get_pronunciation_hint(text: str) -> str:
    """Return a short spoken pronunciation hint for known demo dishes."""
    lower_text = text.lower()
    for pattern, hint in PRONUNCIATION_HINTS:
        if re.search(pattern, lower_text):
            return hint
    return ""


def prepare_tts_text(answer: str, query: str = "") -> str:
    """Convert markdown-heavy assistant text into natural speech."""
    hint = get_pronunciation_hint(f"{query}\n{answer}")

    spoken = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", answer)
    spoken = re.sub(r"`([^`]+)`", r"\1", spoken)
    spoken = re.sub(r"[*_#>•|]+", " ", spoken)
    spoken = re.sub(r"https?://\S+", "", spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()

    try:
        max_chars = int(get_config("ELEVENLABS_MAX_CHARS", "1200"))
    except ValueError:
        max_chars = 1200
    if len(spoken) > max_chars:
        spoken = spoken[:max_chars].rsplit(" ", 1)[0] + "."

    return f"{hint} {spoken}".strip()


def tts_cache_key(text: str, voice_id: str, model_id: str) -> str:
    raw = f"{voice_id}:{model_id}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def generate_elevenlabs_audio(text: str, voice_id: str, model_id: str) -> bytes:
    """Generate MP3 audio with ElevenLabs."""
    api_key = get_config("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ELEVENLABS_API_KEY")

    response = req.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": "mp3_44100_128"},
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.85,
                "style": 0.25,
                "use_speaker_boost": True,
            },
        },
        timeout=30,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "ElevenLabs rejected the API key (401). Check ELEVENLABS_API_KEY and restart the app."
        )
    response.raise_for_status()
    return response.content


def render_tts_control(answer: str, query: str, key: str) -> None:
    """Render an optional ElevenLabs listen button for an assistant answer."""
    if not get_config("ELEVENLABS_API_KEY"):
        return

    voice_id = get_config("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = get_config("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    tts_text = prepare_tts_text(answer, query)
    audio_key = tts_cache_key(tts_text, voice_id, model_id)

    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}

    if st.button("🔊 Listen", key=f"listen_{key}_{audio_key}"):
        with st.spinner("Generating voice..."):
            try:
                st.session_state.audio_cache[audio_key] = generate_elevenlabs_audio(
                    tts_text,
                    voice_id,
                    model_id,
                )
            except Exception as exc:
                st.warning(f"Voice unavailable: {str(exc)[:120]}")

    if audio_key in st.session_state.audio_cache:
        st.audio(st.session_state.audio_cache[audio_key], format="audio/mp3")


def extract_search_term(query: str, answer: str = "") -> str | None:
    """Extract the food/dish/ingredient name from query using LLM."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    result = (llm | StrOutputParser()).invoke(
        f"""Extract the most specific food dish or ingredient name from this cooking context.
    Prefer dish names over individual ingredients.
    If not about food, reply 'NONE'.
    Reply with just the food name, nothing else.

    Examples:
    "How do I make pho?" + "Pho Bo is a Vietnamese..." → "pho"
    "what is nori?" + "Nori is seaweed..." → "nori"
    "weather today?" + "..." → "NONE"
    "how to cook duck coconut" + "Swimming Rama is..." → "Swimming Rama"

    Query: {query}
    Answer: {answer[:200]}
    Food name:"""
    )
    term = result.strip()
    return None if term == "NONE" else term


def get_food_photo(query: str, answer: str = "") -> str | None:
    """Match query to known video thumbnails, fallback to Unsplash."""

    dish_thumbnails = {
        "pho": "https://img.youtube.com/vi/5bIjDYEs6Qc/maxresdefault.jpg",
        "spring roll": "https://img.youtube.com/vi/8CaadFo3sw0/maxresdefault.jpg",
        "goi cuon": "https://img.youtube.com/vi/8CaadFo3sw0/maxresdefault.jpg",
        "bun bo hue": "https://img.youtube.com/vi/qWK_HYlKrAA/maxresdefault.jpg",
        "tom yum": "https://img.youtube.com/vi/ZcGqfJSo5hU/maxresdefault.jpg",
        "duck curry": "https://img.youtube.com/vi/bHgRrOxdFyg/maxresdefault.jpg",
        "thai duck": "https://img.youtube.com/vi/bHgRrOxdFyg/maxresdefault.jpg",
        "duck": "https://img.youtube.com/vi/bHgRrOxdFyg/maxresdefault.jpg",
        "basil chicken": "https://img.youtube.com/vi/q_9rDq2gGmg/maxresdefault.jpg",
        "pad kra pao": "https://img.youtube.com/vi/q_9rDq2gGmg/maxresdefault.jpg",
        "mapo tofu": "https://img.youtube.com/vi/TI2CeY6miDw/maxresdefault.jpg",
        "swimming rama": "https://img.youtube.com/vi/k6NM3lIHCYQ/maxresdefault.jpg",
        "kung pao": "https://img.youtube.com/vi/tjVu_2eQ9SE/maxresdefault.jpg",
        "dumpling": "https://img.youtube.com/vi/LQS_mnNLG3Q/maxresdefault.jpg",
        "char siu": "https://img.youtube.com/vi/By7NwdKdxpE/maxresdefault.jpg",
        "pad thai": "https://img.youtube.com/vi/F86GfZIph8o/maxresdefault.jpg",
    }

    # Check query first, then answer
    for text in [query.lower(), answer.lower()[:300]]:
        for keyword, url in dish_thumbnails.items():
            if re.search(r'\b' + re.escape(keyword) + r's?\b', text):
                return url

    key = get_config("UNSPLASH_ACCESS_KEY")
    if not key:
        return None

    # Fallback to Unsplash only when the API key exists.
    search_term = extract_search_term(query, answer[:200])
    if not search_term:
        return None

    try:
        response = req.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": f"{search_term} food dish asian",
                "per_page": 1,
                "orientation": "landscape"
            },
            headers={"Authorization": f"Client-ID {key}"},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if results:
            return results[0]["urls"]["small"]
        return None
    except Exception:
        return None


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhoBuddy 🍜",
    page_icon="🍜",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header { text-align: center; padding: 1rem 0; }
  .source-box {
    background: #f0f9f0; border-left: 4px solid #4CAF50;
    padding: 0.5rem 1rem; border-radius: 4px;
    font-size: 0.85rem; margin-top: 0.5rem;
  }
  section[data-testid="stSidebar"] > div {
      padding-top: 0rem !important;
      margin-top: -1rem !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 Add a Video")
    new_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    if st.button("📥 Ingest Video", use_container_width=True):
        if new_url:
            with st.spinner("Fetching transcript... (max 15 seconds)"):
                try:
                    result = ingest_single_video(new_url)
                    if result["status"] == "duplicate":
                        st.warning("⏭️ Already in knowledge base!")
                    elif result["status"] == "no_transcript":
                        st.error("❌ No transcript available. Try a different video.")
                    else:
                        st.success(f"✅ Added! {result['chunks']} chunks ingested")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:100]}")
        else:
            st.warning("Please enter a YouTube URL")

    st.markdown("---")
    st.markdown("## 🍽️ Covered Cuisines")
    st.markdown("🇻🇳 **Vietnamese** — Pho, Banh Mi, Spring Rolls")
    st.markdown("🇹🇭 **Thai** — Pad Thai, Green Curry, Tom Yum")
    st.markdown("🇨🇳 **Chinese** — Mapo Tofu, Dumplings, Char Siu")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Try asking")
    demo_prompts = [
        ("🍜 Pho broth", "How do I make pho broth from scratch?"),
        ("🥢 Pad Thai", "Tell me about Pad Thai"),
        ("🍖 Char Siu", "How do I make Char Siu pork?"),
        ("🥟 Dumpling tips", "How do I prevent dumplings from sticking?"),
        ("🔥 Tom Yum", "What makes Tom Yum soup spicy?"),
        ("🌿 Lemongrass swap", "What can I substitute for lemongrass?"),
    ]
    for label, prompt in demo_prompts:
        if st.button(label, use_container_width=True, key=f"ex_{label}", help=prompt):
            st.session_state.pending = prompt

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🍜 PhoBuddy</h1>
  <p>Your AI Asian Cooking Assistant</p>
  <p><em>Vietnamese · Thai · Chinese</em></p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "pending"   not in st.session_state: st.session_state.pending   = None

# ── Load agent (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading PhoBuddy 🍜...")
def load_agent():
    return build_agent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"❌ Could not load agent: {e}")
    st.info("Make sure you've run `python ingest.py` and set your API keys in `.env`")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        photo_url = msg.get("photo_url")
        if photo_url and msg["role"] == "assistant":
            display_image(photo_url)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_tts_control(
                msg["content"],
                msg.get("query", ""),
                msg.get("id", f"history_{idx}"),
            )

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything about Asian cooking... 🥢")

if st.session_state.pending and not user_input:
    user_input = st.session_state.pending
    st.session_state.pending = None

# ── Process ───────────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🍳 Cooking up an answer..."):
            try:
                # Clean query to prevent GPT image refusal
                cleaned_input = clean_query(user_input)

                answer = invoke_agent(
                    agent,
                    cleaned_input,
                    thread_id=st.session_state.thread_id,
                )

                # Use original query for photo matching
                photo_url = get_food_photo(user_input, answer)

                if photo_url:
                    display_image(photo_url)
                st.markdown(answer)
                message_id = str(uuid.uuid4())
                render_tts_control(answer, user_input, message_id)

                st.session_state.messages.append({
                    "id": message_id,
                    "role": "assistant",
                    "content": answer,
                    "photo_url": photo_url,
                    "query": user_input,
                })
            except Exception as e:
                err = f"❌ Something went wrong: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;font-size:0.8rem;'>"
    "PhoBuddy | LangGraph + LCEL + ChromaDB + GPT-4o-mini | Traced by LangSmith"
    "</div>",
    unsafe_allow_html=True,
)
