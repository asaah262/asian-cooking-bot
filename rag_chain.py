"""
rag_chain.py — RAG Chain using LCEL (LangChain Expression Language)
=====================================================================

Pattern:
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )

Memory via RunnableWithMessageHistory.
3 versioned prompt templates for iterations v1 / v2 / v3.
"""

import os
from dotenv import load_dotenv
import chromadb.telemetry.product.posthog as telemetry
telemetry.Posthog.capture = lambda *args, **kwargs: None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
ITERATION      = os.getenv("ITERATION", "v1")

# ── Prompt Templates — one per iteration ──────────────────────────────────────

SYSTEM_V1 = """You are an Asian cooking assistant.
Use the following context from cooking video transcripts to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}
"""

SYSTEM_V2 = """You are an expert Asian cooking assistant specialising in
Vietnamese, Thai, and Chinese cuisine.

Use ONLY the following context from cooking video transcripts to answer.
Always mention which video or channel the information comes from.
If the answer is not in the context, say:
"I don't have that in my cooking video database."

Context:
{context}
"""

SYSTEM_V3 = """You are PhoBuddy 🍜, an expert AI cooking assistant
specialising in Vietnamese, Thai, and Chinese cuisine.

Rules:
- Answer ONLY from the context below — never invent facts
- Always cite the source: "According to [channel] in '[video title]'..."
- For techniques, give clear step-by-step instructions
- For substitutions, suggest authentic alternatives
- If not in context say: "I don't have that in my recipe database."

Cuisine cheat-sheet:
🇻🇳 Vietnamese: fish sauce, lemongrass, fresh herbs, light broths
🇹🇭 Thai: galangal, kaffir lime, coconut milk, chili paste
🇨🇳 Chinese: soy sauce, oyster sauce, wok technique, five spice

Context:
{context}
"""

SYSTEM_PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM_V2, "v3": SYSTEM_V3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_docs(docs):
    """Join retrieved doc chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


# ── Session memory store (in-memory, per session_id) ─────────────────────────

store = {}  # session_id -> ChatMessageHistory

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a ChatMessageHistory for the given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ── RAG chain builder ─────────────────────────────────────────────────────────

def build_rag_chain(iteration: str = None):
    """
    Build LCEL RAG chain with conversation memory.

    Pattern:
      rag_chain = (
          {"context": retriever | format_docs, "question": RunnablePassthrough()}
          | prompt | llm | StrOutputParser()
      )
      wrapped in RunnableWithMessageHistory for memory.

    Returns:
        chain  — call with chain.invoke({"input": q}, config={"configurable": {"session_id": "..."}})
        retriever — for source document display
    """
    iteration = iteration or ITERATION
    print(f"🔧 Building LCEL RAG chain | iteration={iteration}")

    vectorstore = get_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    system_prompt = SYSTEM_PROMPTS.get(iteration, SYSTEM_V1)

    # Window memory: keep last 10 messages (trim_messages pattern from class)
    trimmer = trim_messages(
        max_tokens=2000,
        strategy="last",
        token_counter=llm,
        include_system=True,
    )

    # Prompt with message history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    # Core LCEL chain
    # Note: context is fetched from input["input"] via the retriever
    core_chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs,
            "input":   lambda x: x["input"],
            "history": lambda x: trimmer.invoke(x.get("history", [])) if x.get("history") else [],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with message history for memory
    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    print(f"✅ LCEL RAG chain ready (prompt={iteration})")
    return chain_with_history, retriever


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chain, retriever = build_rag_chain()
    config = {"configurable": {"session_id": "test-session"}}

    questions = [
        "How do I make pho broth from scratch?",
        "What spices did you just mention?",   # tests memory
        "What can I substitute for lemongrass?",
    ]

    print("\n" + "="*60)
    print("🍜 Testing LCEL RAG chain")
    print("="*60)

    for q in questions:
        print(f"\n❓ {q}")
        answer = chain.invoke({"input": q}, config=config)
        print(f"🤖 {answer}")

        # Show source docs
        docs = retriever.invoke(q)
        print("📚 Sources:")
        seen = set()
        for doc in docs:
            title = doc.metadata.get("video_title", "")
            if title not in seen:
                seen.add(title)
                print(f"   - {doc.metadata.get('channel')} | {title}")
        print("-" * 40)
