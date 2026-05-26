"""
app.py — PhoBuddy Streamlit Frontend
=====================================
Uses LangGraph agent (create_react_agent + MemorySaver).
Run with: streamlit run app.py
"""

import os
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

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
                            st.error("❌ No transcript available for this video. Try a different one.")

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
                answer = invoke_agent(
                    agent,
                    user_input,
                    thread_id=st.session_state.thread_id,
                )
                st.markdown(answer)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer
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
