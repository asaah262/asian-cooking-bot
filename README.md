# 🍜 PhoBuddy — Asian Cooking AI Assistant

RAG-powered chatbot that answers questions about Vietnamese, Thai, and Chinese
cooking using YouTube video transcripts.

**Stack**: LangChain · ChromaDB · GPT-4o-mini · LangSmith · Streamlit

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY
#   LANGCHAIN_API_KEY (from smith.langchain.com)
```

### 3. Ingest videos
```bash
# Ingest all curated videos (first run)
python ingest.py

# Add a single video
python ingest.py --single https://youtube.com/watch?v=XXXX

# Reset DB (use between iterations)
python ingest.py --reset
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Iteration Evaluation

### Run evaluation for each iteration:
```bash
# Iteration 1 — baseline
ITERATION=v1 CHUNK_SIZE=1000 CHUNK_OVERLAP=100 python evaluation/evaluate.py

# Iteration 2 — smaller chunks
ITERATION=v2 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python ingest.py --reset
ITERATION=v2 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluation/evaluate.py

# Iteration 3 — better prompt
ITERATION=v3 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluation/evaluate.py
```

Results saved to `evaluations/results_v1.json`, etc.

---

## 🗂️ Project Structure

```
asian_cooking_bot/
├── app.py                  # Streamlit frontend
├── ingest.py               # Transcript fetch + ChromaDB ingestion
├── rag_chain.py            # LangChain RAG pipeline (3 prompt versions)
├── agent.py                # LangChain Agent + 4 tools
├── videos.json             # Curated cooking video list
├── evaluations/
│   └── evaluation.py         # LangSmith evaluation (10 test questions)
├── data/
│   └── chroma_db/          # Persisted vector store (auto-created)
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tools

| Tool | Purpose |
|------|---------|
| `search_cooking_knowledge` | Query ChromaDB vector store |
| `summarize_dish` | Structured dish overview |
| `ingest_new_video` | Add new YouTube video on demand |
| `search_web_for_recipe` | DuckDuckGo fallback |

---

## 📈 Iterations

| | Videos | Chunk Size | Overlap | Prompt | Answer Score |
|-|--------|-----------|---------|--------|-------------|
| **v1** | 9 | 1000 | 100 | Basic | 0.40 |
| **v2** | 13 | 500 | 150 | + Citations | 0.60 |
| **v3** | 13 | 500 | 150 | Full PhoBuddy + Citations | 0.80 |

Key improvements v1 → v3:
- Added 4 new videos (Kung Pao, Dumplings, Char Siu, Pad Thai)
- Smaller chunks = more precise retrieval
- Better prompt = answers cite sources ("According to [channel]...")
- Answer correctness doubled: 0.40 → 0.80

---

## 📹 Video Sources (13 videos, 283 chunks)

- **Helen's Recipes** — Vietnamese cuisine
- **Pailin's Kitchen** — Thai cuisine  
- **Woks of Life** — Chinese cuisine
