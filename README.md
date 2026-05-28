# 🍜 PhoBuddy — Asian Cooking AI Assistant

> Ask me anything about Vietnamese, Thai, and Chinese cooking!

![PhoBuddy Screenshot](https://img.youtube.com/vi/5bIjDYEs6Qc/maxresdefault.jpg)

RAG-powered chatbot that answers questions about Vietnamese, Thai, and Chinese
cooking using YouTube video transcripts. Built with LangChain, LangGraph, ChromaDB,
and deployed on Streamlit Cloud.

**Stack**: LangChain · LangGraph · ChromaDB · GPT-4o-mini · LangSmith · Streamlit · Unsplash

🌐 **Live App**: [PhoBuddy on Streamlit Cloud](https://asian-cooking-bot-cpkc3vyb9bkw8ktdtdxhvn.streamlit.app/)

---

## ✨ Features

- 🍜 Ask cooking questions in natural language
- 📸 Automatic food photos with every answer
- 🔍 RAG retrieval from 15 YouTube cooking videos
- 🧠 Conversation memory across turns
- 🌐 Web search fallback for unknown dishes
- 📥 Add new YouTube videos on the fly
- 📊 Evaluated with LangSmith, ROUGE, and Giskard

---

## 🍽️ Covered Cuisines

| Cuisine | Dishes |
|---------|--------|
| 🇻🇳 Vietnamese | Pho Bo, Bun Bo Hue, Fresh Spring Rolls |
| 🇹🇭 Thai | Tom Yum, Duck Curry, Basil Chicken, Swimming Rama, Pad Thai |
| 🇨🇳 Chinese | Mapo Tofu, Kung Pao Chicken, Dumplings, Char Siu |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
python -m pip install --upgrade -r requirements.txt
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your API keys:
#   OPENAI_API_KEY
#   LANGCHAIN_API_KEY (from smith.langchain.com)
#   UNSPLASH_ACCESS_KEY (from unsplash.com/developers)
```

### 3. Ingest videos
```bash
# Ingest all curated videos (first run)
python ingest.py

# Repair Chroma metadata from videos.json without re-embedding
python ingest.py --sync-metadata

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

## 🗂️ Project Structure

```
asian_cooking_bot/
├── app.py                  # Streamlit frontend
├── ingest.py               # Transcript fetch + ChromaDB ingestion
├── rag_chain.py            # LangChain RAG pipeline (v1-v5 prompts)
├── agent.py                # LangGraph Agent + 4 tools
├── videos.json             # Curated cooking video list (15 videos)
├── evaluations/
│   └── evaluation.py       # LangSmith + ROUGE + Giskard evaluation
├── data/
│   └── chroma_db/          # Persisted vector store (392 chunks)
├── requirements.txt
└── .env.example
```

---

## 🏗️ Architecture

```
User Question (text)
        ↓
   Streamlit UI (app.py)
        ↓
   LangGraph ReAct Agent (agent.py)
   + MemorySaver (conversation history)
        ↓
   ┌─────────────────────────────────┐
   │  Tool 1: search_cooking_knowledge│ ← LCEL RAG chain → ChromaDB
   │  Tool 2: summarize_dish          │ ← structured dish overview
   │  Tool 3: ingest_new_video        │ ← add YouTube video on demand
   │  Tool 4: search_web_for_recipe   │ ← DuckDuckGo fallback
   └─────────────────────────────────┘
        ↓
   GPT-4o-mini → Answer + Sources
        ↓
   Unsplash API → Food Photo
        ↓
   User sees: Photo + Answer 
```

---

## 🛠️ Tools

| Tool | Purpose |
|------|---------|
| `search_cooking_knowledge` | LCEL RAG chain → ChromaDB vector store |
| `summarize_dish` | Structured dish overview (🍽️ ingredients, method, tips) |
| `ingest_new_video` | Fetch transcript + embed new YouTube video |
| `search_web_for_recipe` | DuckDuckGo fallback when not in DB |

---

## 📊 Iteration Evaluation

### Run evaluation for each iteration:
```bash
# Iteration 1 — baseline
ITERATION=v1 CHUNK_SIZE=1000 CHUNK_OVERLAP=100 python evaluations/evaluation.py

# Iteration 2 — more videos + smaller chunks
ITERATION=v2 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python ingest.py --reset
ITERATION=v2 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluations/evaluation.py

# Iteration 3 — better prompt + citations
ITERATION=v3 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluations/evaluation.py

# Iteration 4 — web search fallback + tuned NOT_IN_DATABASE
ITERATION=v4 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluations/evaluation.py

# Iteration 5 — grounded citations + clearer agent tool routing
ITERATION=v5 CHUNK_SIZE=500 CHUNK_OVERLAP=150 python evaluations/evaluation.py

# Giskard auto-evaluation (v4)
ITERATION=v4 python evaluations/evaluation.py --giskard
```

---

## 📈 Iterations

| | Videos | Chunk Size | Overlap | Prompt | Answer Score | ROUGE-1 |
|-|--------|-----------|---------|--------|-------------|---------|
| **v1** | 9 | 1000 | 100 | Basic | 0.40 | 0.147 |
| **v2** | 13 | 500 | 150 | + Citations | 0.60 | 0.086 |
| **v3** | 13 | 500 | 150 | Full PhoBuddy | 0.80 | 0.093 |
| **v4** | 13 | 500 | 150 | + Web fallback | 0.60 | **0.292** |
| **v5** | 15 | 500 | 150 | + Grounded citations & tool routing | 0.80 | **0.295** |

## 📸 Screenshots

Evaluation screenshots organized by iteration:

```
screenshots/
├── iteration1/          # v1 baseline (answer_score: 0.40)
├── iteration2/          # v2 +4 videos, smaller chunks (0.60)
├── iteration3/          # v3 PhoBuddy prompt (0.80)
├── iteration4/          # v4 web search fallback (0.60, ROUGE: 0.206)
├── iteration4-betterQA/ # v4 with corrected reference answers (ROUGE: 0.292)
├── iteration4-giskard/  # v4 Giskard auto-evaluation
└── iteration5/          # v5 grounded citations + tool routing (ROUGE: 0.295)
```

### Giskard Auto-Evaluation (v4)

| Question Type | Correctness |
|---------------|-------------|
| Simple | 75% |
| Conversational | 67% |
| Double | 67% |
| Situational | 67% |
| Complex | 50% |
| **Distracting element** | **100%** |

---

## 📹 Video Sources (15 videos, 392 chunks)

**🇻🇳 Helen's Recipes** — Vietnamese cuisine
| Video | Chunks |
|-------|--------|
| Pho Bo - Vietnamese Beef Noodle Soup | 13 |
| Fresh Spring Roll (GOI CUON) | 8 |
| Bun Bo Hue - Vietnamese Spicy Beef Noodle Soup | 17 |

**🇹🇭 Pailin's Kitchen** — Thai & Chinese cuisine
| Video | Chunks |
|-------|--------|
| Tom Yum Goong Recipe (Creamy Style!) | 32 |
| Thai Duck Curry | 49 |
| Thai Basil Chicken - Pad Kra Pao | 24 |
| Tom Yum Soup - Authentic Thai | 50 |
| Mapo Tofu | 24 |
| Swimming Rama | 23 |
| Pad Thai | 30 |

**🇨🇳 Chinese cuisine**
| Video | Chunks |
|-------|--------|
| Kung Pao Chicken | 37 |
| Dumplings - Dim Sum | 5 |
| Char Siu - Chinese BBQ Pork | 34 |

**➕ Additional v5 videos**
| Video | Chunks |
|-------|--------|
| User-added (nSgkcDCG6ck) | 23 |
| User-added (UgreUS9-Sig) | 23 |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key |
| `LANGCHAIN_API_KEY` | ✅ | LangSmith API key |
| `LANGCHAIN_TRACING_V2` | ✅ | Enable LangSmith tracing |
| `LANGCHAIN_PROJECT` | ✅ | LangSmith project name |
| `CHROMA_DB_PATH` | ✅ | Path to ChromaDB |
| `CHUNK_SIZE` | ✅ | Chunk size for text splitting |
| `CHUNK_OVERLAP` | ✅ | Chunk overlap for text splitting |
| `ITERATION` | ✅ | Prompt version (v1/v2/v3/v4/v5) |
| `UNSPLASH_ACCESS_KEY` | 🟡 Optional | Food photos |
