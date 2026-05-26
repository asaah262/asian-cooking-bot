"""
app.py — PhoBuddy Streamlit Frontend
=====================================
Uses LangGraph agent (create_react_agent + MemorySaver).
Run with: streamlit run app.py
"""

import os
import uuid
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Helper functions ──────────────────────────────────────────────────────────

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


def extract_search_term(query: str, answer: str = "") -> str | None:
    """Extract the food/dish/ingredient name from query using LLM."""
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

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

    # Fallback to Unsplash
    search_term = extract_search_term(query, answer[:200])
    if not search_term:
        return None

    import requests as req
    key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not key:
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
        data = response.json()
        if data["results"]:
            return data["results"][0]["urls"]["small"]
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
                    from ingest import fetch_transcript, build_documents, extract_video_id
                    from langchain_openai import OpenAIEmbeddings
                    from langchain_chroma import Chroma
                    import json

                    video_id = extract_video_id(new_url)

                    # Check duplicate
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vectorstore = Chroma(
                        collection_name="asian_cooking",
                        embedding_function=embeddings,
                        persist_directory="./data/chroma_db",
                    )
                    existing = vectorstore.get()
                    existing_ids = {m.get("video_id") for m in existing["metadatas"] if m}

                    if video_id in existing_ids:
                        st.warning("⏭️ Already in knowledge base!")
                    else:
                        transcript = fetch_transcript(new_url)
                        if transcript:
                            video_meta = {
                                "url": new_url,
                                "title": f"User-added ({video_id})",
                                "channel": "User-added",
                                "cuisine": "Asian",
                                "tags": [],
                            }
                            docs = build_documents(video_meta, transcript)
                            vectorstore.add_documents(docs)

                            # Auto-save to videos.json
                            try:
                                with open("videos.json", "r") as f:
                                    existing_videos = json.load(f)
                                if new_url not in [v["url"] for v in existing_videos]:
                                    existing_videos.append(video_meta)
                                    with open("videos.json", "w") as f:
                                        json.dump(existing_videos, f, indent=2)
                            except Exception:
                                pass

                            st.success(f"✅ Added! {len(docs)} chunks ingested")
                        else:
                            st.error("❌ No transcript available. Try a different video.")

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
    st.markdown("### 💡 Try asking:")
    examples = [
        "How do I make pho broth?",
        "What's in Pad Thai?",
        "How do I fold dumplings?",
        "Substitute for fish sauce?",
        "Tell me about Mapo Tofu",
    ]
    for eq in examples:
        if st.button(eq, use_container_width=True, key=f"ex_{eq}"):
            st.session_state.pending = eq

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
    from agent import build_agent
    return build_agent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"❌ Could not load agent: {e}")
    st.info("Make sure you've run `python ingest.py` and set your API keys in `.env`")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        photo_url = msg.get("photo_url")
        if photo_url and msg["role"] == "assistant":
            st.image(photo_url, use_column_width=True)
        st.markdown(msg["content"])

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
                from agent import invoke_agent

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
                    st.image(photo_url, use_column_width=True)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "photo_url": photo_url,
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