"""
agent.py — LangChain Agent using LangGraph
=================================================================

Pattern:
  from langgraph.prebuilt import create_react_agent
  from langgraph.checkpoint.memory import MemorySaver

  memory = MemorySaver()
  agent  = create_react_agent(llm, tools, checkpointer=memory)
  response = agent.invoke(
      {"messages": [("user", query)]},
      config={"configurable": {"thread_id": "session-1"}}
  )
"""

import os
from dotenv import load_dotenv
import chromadb.telemetry.product.posthog as telemetry
telemetry.Posthog.capture = lambda *args, **kwargs: None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
ITERATION      = os.getenv("ITERATION", "v3")


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── Tool 1: RAG Search ────────────────────────────────────────────────────────
# Pattern: @tool decorator
# RAG chain: LCEL pattern

@tool
def search_cooking_knowledge(query: str) -> str:
    """
    Search the Asian cooking video knowledge base for recipes, techniques,
    ingredients, and tips. Use this for ANY question about Vietnamese, Thai,
    or Chinese cooking. Input should be a clear cooking question or topic.
    """
    vectorstore = get_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # LCEL RAG chain
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an Asian cooking expert. Answer based ONLY on the context below.
        Always mention which channel or video the information comes from.
        If not in context, say "I don't have that in my cooking video database."

        Context:
        {context}

        Question: {question}
        """
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(query)

    # Append source citations
    docs = retriever.invoke(query)
    seen, sources = set(), []
    for doc in docs:
        title = doc.metadata.get("video_title", "")
        if title not in seen:
            seen.add(title)
            sources.append(
                f"  • {doc.metadata.get('channel')} — {doc.metadata.get('video_title')}"
            )

    if sources:
        answer += "\n\n📚 Sources:\n" + "\n".join(sources)

    return answer


# ── Tool 2: Dish Summary ──────────────────────────────────────────────────────

@tool
def summarize_dish(dish_name: str) -> str:
    """
    Get a structured summary of a dish including what it is, key ingredients,
    cooking method, and pro tips. Use when user asks 'what is X?' or
    'tell me about X'. Input: dish name like 'Pho', 'Pad Thai', 'Mapo Tofu'.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(dish_name, k=6)

    if not docs:
        return f"I don't have information about {dish_name} in my knowledge base yet."

    context = "\n\n".join([d.page_content for d in docs])
    sources = list({d.metadata.get("video_title", "") for d in docs})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = f"""Based on these cooking video transcripts, give a structured
    summary of {dish_name}:

    {context}

    Format:
    🍽️ What is it: (1-2 sentences)
    🥘 Key ingredients: (bullet list)
    👨‍🍳 Cooking method: (brief overview)
    ⏱️ Typical cook time: (if mentioned)
    💡 Pro tip: (one key tip from the videos)
    """
    response = llm.invoke(prompt)
    return response.content + f"\n\n📹 Covered in: {', '.join(sources)}"


# ── Tool 3: Ingest New Video ──────────────────────────────────────────────────

@tool
def ingest_new_video(youtube_url: str) -> str:
    """
    Fetch the transcript of a new YouTube cooking video and add it to the
    knowledge base so you can answer questions about it.
    Use ONLY when the user provides a YouTube URL.
    Input: a valid YouTube URL.
    """
    import re
    import time
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    from youtube_transcript_api import YouTubeTranscriptApi

    # Extract video ID
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    ]
    video_id = None
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        return f"❌ Invalid YouTube URL: {youtube_url}"

    # Check if already in DB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    existing = vectorstore.get()
    existing_ids = {m.get("video_id") for m in existing["metadatas"] if m}
    if video_id in existing_ids:
        return f"⏭️ This video is already in the knowledge base! (ID: {video_id})"

    # Fetch transcript
    try:
        ytt_api = YouTubeTranscriptApi()
        try:
            fetched = ytt_api.fetch(video_id, languages=['en'])
        except Exception:
            fetched = ytt_api.fetch(video_id)
        full_text = " ".join([entry.text for entry in fetched])
        # Clean transcript
        full_text = re.sub(r'\[.*?\]', '', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
    except Exception as e:
        return f"❌ Could not fetch transcript: {e}"

    # Chunk and ingest
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
    )
    chunks = splitter.split_text(full_text)
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "video_id":    video_id,
                "video_url":   youtube_url,
                "video_title": f"User-added video ({video_id})",
                "channel":     "User-added",
                "cuisine":     "Asian",
                "tags":        "",
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
        for i, chunk in enumerate(chunks)
    ]
    vectorstore.add_documents(documents)

    return (
        f"✅ Video ingested successfully! Added {len(chunks)} chunks.\n"
        f"Video ID: {video_id}\n"
        f"You can now ask questions about this video!"
    )


# ── Tool 4: Web Search Fallback ───────────────────────────────────────────────

@tool
def search_web_for_recipe(query: str) -> str:
    """
    Search the web for Asian cooking information NOT found in the video
    knowledge base. Use as a LAST RESORT when RAG search returns nothing useful.
    Input: a specific cooking question.
    """
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        return "🌐 Web search results:\n" + search.run(query + " Asian cooking recipe")
    except Exception as e:
        return f"Web search unavailable: {e}"


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent():
    """
    Build a LangGraph ReAct agent with 4 tools and MemorySaver.

    Usage:
        agent  = build_agent()
        config = {"configurable": {"thread_id": "session-1"}}
        resp   = agent.invoke({"messages": [("user", "How do I make pho?")]}, config)
        print(resp["messages"][-1].content)
    """
    tools = [
        search_cooking_knowledge,
        summarize_dish,
        ingest_new_video,
        search_web_for_recipe,
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # System prompt passed directly at agent creation
    system_prompt = """You are PhoBuddy 🍜, an expert AI cooking assistant
        specialising in Vietnamese, Thai, and Chinese cuisine.

        How to use your tools:
            - ALWAYS try search_cooking_knowledge first for any cooking question
            - Use summarize_dish when the user asks "what is X?" or "tell me about X"
            - Use ingest_new_video ONLY when the user gives you a YouTube URL
            - Use search_web_for_recipe ONLY as last resort if RAG returns nothing useful

        Be friendly, specific, and always cite your video sources."""

    # MemorySaver for conversation history
    memory = MemorySaver()

    # LangGraph create_react_agent
    agent = create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt,
        checkpointer=memory,
    )

    print("✅ LangGraph agent ready (4 tools + MemorySaver)")
    return agent


def invoke_agent(agent, user_input: str, thread_id: str = "default") -> str:
    """
    Helper to invoke the agent and extract the final text response.

    Args:
        agent:      built agent from build_agent()
        user_input: the user's message
        thread_id:  conversation thread (same thread = shared memory)

    Returns:
        str: the agent's text response
    """
    config   = {"configurable": {"thread_id": thread_id}}
    response = agent.invoke(
        {"messages": [("user", user_input)]},
        config=config,
    )
    # Last message is the agent's final reply
    return response["messages"][-1].content


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🍜 PhoBuddy — Starting LangGraph agent...")
    agent = build_agent()

    questions = [
        "Tell me about Pad Thai",
        "What are the key spices in Vietnamese pho?",
        "What can I use instead of fish sauce?",
        "What did I just ask you about?",      # tests MemorySaver
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"❓ {q}")
        answer = invoke_agent(agent, q, thread_id="test-session")
        print(f"🤖 {answer}")
