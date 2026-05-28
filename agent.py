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
import uuid
from dotenv import load_dotenv
import chromadb.telemetry.product.posthog as telemetry
telemetry.Posthog.capture = lambda *args, **kwargs: None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from ingest import ingest_single_video
from rag_chain import build_rag_chain, format_docs
from langchain_community.tools import DuckDuckGoSearchRun


load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
ITERATION      = os.getenv("ITERATION", "v5")


# ── Shared helpers ────────────────────────────────────────────────────────────

def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


# ── Tool 1: RAG Search ────────────────────────────────────────────────────────
# Pattern: @tool decorator
# RAG chain: LCEL pattern

@tool
def search_cooking_knowledge(query: str) -> str:
    """
    Search for specific cooking details: techniques, substitutions, timing,
    temperatures, ingredient amounts, troubleshooting, or step-by-step recipe
    questions. Do NOT use for general dish overview questions like "what is X",
    "tell me about X", "show me X", or "give me an overview of X"; use
    summarize_dish for those first.
    """
    
    chain, retriever = build_rag_chain(iteration=ITERATION)
    config = {"configurable": {"session_id": f"rag-tool-{uuid.uuid4()}"}}
    answer = chain.invoke({"input": query}, config=config)

    # Check if RAG found nothing relevant
    if "NOT_IN_DATABASE" in answer or "don't have that in my recipe database" in answer:
        return "I don't have that in my cooking video database. Please use web search."

    # Append sources
    docs = retriever.invoke(query)
    seen, sources = set(), []
    for doc in docs:
        title = doc.metadata.get("video_title", "")
        if title not in seen:
            seen.add(title)
            sources.append(
                f"  • {doc.metadata.get('channel')} — {title}"
            )
    if sources:
        answer += "\n\n📚 Sources:\n" + "\n".join(sources)

    return answer


# ── Tool 2: Dish Summary ──────────────────────────────────────────────────────

@tool
def summarize_dish(dish_name: str) -> str:
    """
    Get a structured overview of a known dish: what it is, key ingredients,
    cooking method, cook time, and pro tips. Use this FIRST for general dish
    overview questions such as "what is X", "tell me about X", "show me X",
    "describe X", or "give me an overview of X". Do not use for step-by-step
    recipe, substitution, timing, temperature, or troubleshooting questions.
    Input should be only the dish name.
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(dish_name, k=6)

    if not docs:
        return "NOT_IN_DATABASE"

    # Check relevance — docs must actually mention the dish
    dish_lower = dish_name.lower()
    relevant_docs = [
        d for d in docs
        if dish_lower in d.page_content.lower()
        or dish_lower in d.metadata.get("video_title", "").lower()
    ]

    if not relevant_docs:
        return "NOT_IN_DATABASE"

    context = format_docs(relevant_docs)
    sources = sorted({
        f"{d.metadata.get('channel', 'Unknown channel')} — "
        f"{d.metadata.get('video_title', 'Unknown video')}"
        for d in relevant_docs
    })

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    prompt = f"""Based only on these cooking video transcript excerpts, give a
    structured summary of {dish_name}. Use the source lines exactly when citing:

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
    try:
        result = ingest_single_video(youtube_url)
    except Exception as e:
        return f"❌ Could not ingest video: {e}"

    if result["status"] == "duplicate":
        return f"⏭️ This video is already in the knowledge base! (ID: {result['video_id']})"

    if result["status"] == "no_transcript":
        return "❌ Could not fetch a transcript for this video."

    return (
        f"✅ Video ingested successfully! Added {result['chunks']} chunks.\n"
        f"Video ID: {result['video_id']}\n"
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
        summarize_dish,
        search_cooking_knowledge,
        ingest_new_video,
        search_web_for_recipe,
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # System prompt passed directly at agent creation
    system_prompt = """You are PhoBuddy 🍜, an Asian cooking assistant.

    Your knowledge base covers these dishes:
    - Vietnamese: Pho Bo, Bun Bo Hue, Fresh Spring Rolls (Goi Cuon)
    - Thai: Tom Yum, Duck Curry, Basil Chicken, Swimming Rama, Pad Thai
    - Chinese: Mapo Tofu, Kung Pao Chicken, Dumplings, Char Siu BBQ Pork

    Tool routing rules:
        - summarize_dish → MUST use first for general dish overview questions:
        "what is X?", "tell me about X", "show me X", "describe X",
        "give me an overview of X"
        - search_cooking_knowledge → MUST use for procedural or specific
        cooking questions: "how do I make/cook/prepare X", "how do I prevent X",
        "substitute for X", "how long", "what temperature", "what spices",
        ingredient quantities, step-by-step recipes, troubleshooting
        - ingest_new_video → ONLY when user provides a YouTube URL
        - search_web_for_recipe → ONLY when a tool returns NOT_IN_DATABASE

    CRITICAL rules:
    - Never mention images or photos in responses
    - Never say "I can't provide images" or "I can't show photos"
    - Answer exactly what was asked — be concise
    - If not about food at all, answer directly without tools"""

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
