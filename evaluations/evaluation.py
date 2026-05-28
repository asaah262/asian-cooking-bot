"""
evaluation/evaluate.py — RAG Evaluation with LangSmith
=======================================================

Evaluation flow (3 types — as taught in class):
  Type 1: Answer correctness   (LLM-as-judge, 0/1 score)
  Type 2: Answer hallucination (does answer match context?)
  Type 3: ROUGE score          (n-gram overlap with reference)

Usage:
  ITERATION=v1 python evaluation/evaluate.py
  ITERATION=v2 python evaluation/evaluate.py
  ITERATION=v3 python evaluation/evaluate.py
  ITERATION=v5 python evaluation/evaluate.py
"""

import os
import sys
import json
import argparse
import importlib
from datetime import datetime
from langsmith import Client, traceable
from langsmith.evaluation import evaluate as ls_evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()
# LangSmith auto-traces when env vars set
os.environ["LANGCHAIN_TRACING_V2"] = "true"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent import build_agent, invoke_agent
from rag_chain import format_doc

ITERATION     = os.getenv("ITERATION", "v5")
CHUNK_SIZE    = os.getenv("CHUNK_SIZE", "1000")
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", "100")
CHROMA_PATH   = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
DATASET_NAME  = "asian-cooking-bot-eval"
_EVAL_AGENT   = None

# ── 10 fixed test Q&A pairs (same across all iterations for fair comparison) ──

QA_PAIRS = [
    {
        "question": "How do I make pho broth from scratch?",
        "answer": "Start by blanching beef bones in boiling water for 5-10 minutes, rinse well, then simmer with charred ginger, onion, star anise, cinnamon sticks, and black cardamom for 1-2 hours. Season with fish sauce, rock sugar, and salt."
    },
    {
        "question": "What are the key ingredients in Pad Thai?",
        "answer": "Pad Thai uses rice noodles, tamarind paste, fish sauce, eggs, bean sprouts, and peanuts."
    },
    {
        "question": "How do I make Mapo Tofu?",
        "answer": "Mapo Tofu uses soft tofu, ground beef or pork, Sichuan peppercorns, soy sauce, and chili oil. Saute the meat, add tofu gently, simmer and thicken with potato starch."
    },
    {
        "question": "What can I substitute for lemongrass?",
        "answer": "If you can't find fresh lemongrass, use lime zest or dried lemongrass. Use more dried lemongrass as it has less concentrated flavor."
    },
    {
        "question": "How long should I simmer pho bones?",
        "answer": "After blanching and rinsing the bones, simmer the pho broth for about 1 to 2 hours over medium heat, skimming regularly."
    },
    {
        "question": "How do I make Char Siu pork?",
        "answer": "Marinate pork with soy sauce, honey, hoisin sauce, and Chinese five spice, then roast until caramelized and slightly charred on the outside."
    },
    {
        "question": "What vegetables are used in Vietnamese spring rolls?",
        "answer": "Vietnamese spring rolls use spring onions, Asian Thai basil, bean sprouts, cucumber, and herbs like sawtooth coriander, wrapped in rice paper."
    },
    {
        "question": "How do I make dumpling dough?",
        "answer": "Mix 50g wheat starch and 25g potato starch with 70g boiling water, knead until smooth, then rest before rolling thin."
    },
    {
        "question": "What makes Tom Yum soup spicy?",
        "answer": "Tom Yum gets its spiciness from fresh Thai chilies, which are pounded in a mortar to distribute heat evenly throughout the soup."
    },
    {
        "question": "How do I cook Kung Pao chicken?",
        "answer": "Stir fry chicken with dried chilies, Sichuan peppercorns, peanuts, and a sauce of soy sauce, vinegar, and sugar in a hot wok."
    },
]


# ── RAG pipeline with @traceable ─────────────────────────────────────────────

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="asian_cooking",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n---\n\n".join(format_doc(doc) for doc in docs)


def get_eval_agent():
    """Build the same agent used by the Streamlit app once per evaluation run."""
    global _EVAL_AGENT
    if _EVAL_AGENT is None:
        _EVAL_AGENT = build_agent()
    return _EVAL_AGENT


@traceable()
def retrieve_docs(question: str):
    """Retrieve relevant docs — traced in LangSmith."""
    retriever = get_retriever()
    return retriever.invoke(question)


@traceable()
def get_answer(question: str) -> dict:
    """
    Production app path — traced in LangSmith.
    Returns {"answer": str, "contexts": list[str]}
    """
    docs = retrieve_docs(question)
    answer = invoke_agent(
        get_eval_agent(),
        question,
        thread_id=f"eval-{ITERATION}-{abs(hash(question))}",
    )

    return {
        "answer":   answer,
        "contexts": [format_doc(doc) for doc in docs],
    }


# ── Predict function for LangSmith evaluate() ────────────────────────────────

def predict_rag_answer(example: dict) -> dict:
    """Wrapper called by LangSmith evaluate() for each example."""
    result = get_answer(example["question"])
    return {"answer": result["answer"]}


def predict_rag_answer_with_context(example: dict) -> dict:
    """Wrapper that also returns context (for hallucination eval)."""
    result = get_answer(example["question"])
    return {"answer": result["answer"], "contexts": result["contexts"]}


# ── Evaluators (LLM-as-judge) ─────────────────────────────────────────────────

def answer_evaluator(run, example) -> dict:
    """
    Type 1: Answer correctness.
    LLM-as-judge scores 0 or 1.
    """
    input_question  = example.inputs["question"]
    reference       = example.outputs["answer"]
    prediction      = run.outputs["answer"]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a teacher grading a cooking chatbot answer.
Score 1 if the answer is correct and useful for a home cook.
Score 0 if it is wrong, irrelevant, or says "I don't know" when it shouldn't.
Return ONLY the number 0 or 1."""),
        ("human", """Question: {question}
Reference answer: {reference}
Student answer: {prediction}

Score (0 or 1):"""),
    ])

    grader = grader_prompt | llm | StrOutputParser()
    score_str = grader.invoke({
        "question":   input_question,
        "reference":  reference,
        "prediction": prediction,
    })

    try:
        score = int(score_str.strip())
    except Exception:
        score = 0

    return {"key": "answer_score", "score": score}


def hallucination_evaluator(run, example) -> dict:
    """
    Type 2: Hallucination check.
    Does the answer stick to the retrieved context?
    """
    prediction = run.outputs.get("answer", "")
    contexts   = run.outputs.get("contexts", [])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are checking if a cooking chatbot hallucinated.
Score 1 if the answer is fully supported by the documents below.
Score 0 if the answer contains facts NOT in the documents.
Return ONLY 0 or 1."""),
        ("human", """Documents:
{context}

Answer: {answer}

Score (0 = hallucination, 1 = grounded):"""),
    ])

    grader = grader_prompt | llm | StrOutputParser()
    score_str = grader.invoke({
        "context": "\n\n".join(contexts[:3]),
        "answer":  prediction,
    })

    try:
        score = int(score_str.strip())
    except Exception:
        score = 1  # default to no hallucination if parse fails

    return {"key": "answer_hallucination", "score": score}


# ── ROUGE evaluation ──────────────────────────────────────────────────────────

def run_rouge_evaluation(results: list[dict]) -> dict:
    """
    Compute ROUGE scores comparing generated answers to references.
    Uses HuggingFace evaluate library — as taught in class.
    """
    try:
        hf_evaluate = importlib.import_module('evaluate')
        rouge = hf_evaluate.load("rouge")

        predictions = [r["answer"] for r in results if "answer" in r]
        references  = [r["reference"] for r in results if "reference" in r]

        if not predictions:
            return {}

        scores = rouge.compute(predictions=predictions, references=references)
        print(f"\n📊 ROUGE Scores:")
        print(f"   ROUGE-1: {round(scores['rouge1'], 3)}")
        print(f"   ROUGE-2: {round(scores['rouge2'], 3)}")
        print(f"   ROUGE-L: {round(scores['rougeL'], 3)}")
        return scores

    except ImportError:
        print("⚠️  Install `evaluate` and `rouge_score` for ROUGE: pip install evaluate rouge_score")
        return {}


# ── LangSmith dataset setup ───────────────────────────────────────────────────

def setup_langsmith_dataset() -> str:
    """Create the evaluation dataset in LangSmith (once)."""
    client = Client()

    # Check if dataset already exists
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"📋 Dataset '{DATASET_NAME}' already exists in LangSmith")
        return DATASET_NAME

    print(f"📋 Creating dataset '{DATASET_NAME}' in LangSmith...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="10 Asian cooking Q&A pairs for RAG evaluation",
    )

    client.create_examples(
        inputs=[{"question": qa["question"]} for qa in QA_PAIRS],
        outputs=[{"answer": qa["answer"]} for qa in QA_PAIRS],
        dataset_id=dataset.id,
    )

    print(f"✅ Created {len(QA_PAIRS)} examples in LangSmith")
    return DATASET_NAME


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation():
    print(f"\n{'='*60}")
    print(f"🧪 Evaluation | iteration={ITERATION} | chunk={CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"{'='*60}\n")

    # 1. Setup dataset in LangSmith
    dataset_name = setup_langsmith_dataset()

    # 2. Run LangSmith evaluation — Type 1: Answer correctness
    print("\n🔬 Running Type 1: Answer correctness...")
    experiment_correctness = ls_evaluate(
        predict_rag_answer,
        data=dataset_name,
        evaluators=[answer_evaluator],
        experiment_prefix=f"cooking-bot-{ITERATION}",
        max_concurrency=1,
        metadata={
            "iteration":     ITERATION,
            "chunk_size":    CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )

    # 3. Run LangSmith evaluation — Type 2: Hallucination
    print("\n🔬 Running Type 2: Hallucination check...")
    experiment_hallucination = ls_evaluate(
        predict_rag_answer_with_context,
        data=dataset_name,
        evaluators=[hallucination_evaluator],
        experiment_prefix=f"cooking-bot-{ITERATION}-hallucination",
        max_concurrency=1,
        metadata={"iteration": ITERATION},
    )

    # 4. ROUGE evaluation (local)
    print("\n🔬 Running ROUGE evaluation...")
    local_results = []
    for qa in QA_PAIRS:
        result = get_answer(qa["question"])
        local_results.append({
            "question":  qa["question"],
            "answer":    result["answer"],
            "reference": qa["answer"],
        })

    rouge_scores = run_rouge_evaluation(local_results)

    # 5. Save summary
    summary = {
        "iteration":     ITERATION,
        "chunk_size":    CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "timestamp":     datetime.now().isoformat(),
        "rouge_scores":  rouge_scores,
        "langsmith_experiments": {
            "correctness":   f"cooking-bot-{ITERATION}",
            "hallucination": f"cooking-bot-{ITERATION}-hallucination",
        },
        "langsmith_url": "https://smith.langchain.com",
    }

    os.makedirs("evaluations", exist_ok=True)
    out_file = f"evaluations/results_{ITERATION}.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "details": local_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete — {ITERATION}")
    print(f"📊 ROUGE-1: {rouge_scores.get('rouge1', 'N/A')}")
    print(f"🔗 Full results in LangSmith: https://smith.langchain.com")
    print(f"💾 Local results saved: {out_file}")
    print(f"{'='*60}")

    return summary


# ── Optional: Giskard auto-testset ───────────────────────────────────────────
# Run this separately for v3 — generates questions automatically!

def run_giskard_evaluation():
    """
    Auto-generate test questions from ChromaDB and evaluate the agent.
    Run for iteration v4 to get an impressive auto-evaluation.
    """
    try:
        from giskard.rag import KnowledgeBase, generate_testset, evaluate as gsk_evaluate
        import pandas as pd
        import pickle

        print("\n🤖 Running Giskard auto-evaluation...")

        # Build knowledge base from ChromaDBn

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            collection_name="asian_cooking",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        all_docs = vectorstore.get()
        df_giskard = pd.DataFrame(
            [{"text": doc} for doc in all_docs["documents"]],
            columns=["text"]
        )

        kb_giskard = KnowledgeBase(df_giskard)

        test_questions = generate_testset(
            kb_giskard,
            num_questions=20,
            agent_description="Asian cooking assistant for Vietnamese, Thai and Chinese recipes.",
        )

        # Save kb to avoid re-spending tokens
        pickle.dump(kb_giskard, open('evaluations/kb_giskard.pkl', 'wb'))

        # Evaluation function
        def use_agent(question, history=None):
            result = get_answer(question)
            return result["answer"]

        report = gsk_evaluate(
            use_agent,
            testset=test_questions,
            knowledge_base=kb_giskard,
        )

        print("✅ Giskard evaluation complete!")
        print(report.correctness_by_question_type())
        return report

    except ImportError:
        print("⚠️  Giskard not installed. Run: pip install 'giskard[llm]'")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--giskard", action="store_true",
                        help="Also run Giskard auto-evaluation (v3 only)")
    args = parser.parse_args()

    run_evaluation()

    if args.giskard:
        run_giskard_evaluation()
