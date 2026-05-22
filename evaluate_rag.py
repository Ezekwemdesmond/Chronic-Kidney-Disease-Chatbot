"""
CKD Chatbot RAG Pipeline — RAGAS Evaluation Script
===================================================
Evaluates the hybrid-search RAG chatbot using four RAGAS metrics:

  Metric                       What it measures
  ─────────────────────────    ─────────────────────────────────────────
  Context Precision            Fraction of retrieved chunks relevant to the query
  Context Recall               Fraction of ground-truth info covered by chunks
  Faithfulness                 Answer stays within retrieved context (no hallucination)
  Answer Relevancy             Answer addresses what was actually asked

Medical questions are scored on all four metrics.
Conversational questions (greetings, off-topic) are scored on Answer Relevancy only.

Usage
-----
  python evaluate_rag.py                    # Full run (20 questions)
  python evaluate_rag.py --sample 5         # Quick smoke-test
  python evaluate_rag.py --category factual_medical
  python evaluate_rag.py --top-k 8         # More retrieved chunks

Output
------
  Console : per-metric scores for medical and conversational sets
  data/   : eval_rag_results_raw.json  (questions + answers, for manual review)
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8")

import ssl
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

def _ssl_context(**kwargs):
    kwargs.setdefault("cafile", certifi.where())
    return ssl.create_default_context(**kwargs)

ssl._create_default_https_context = _ssl_context

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# RAG pipeline imports (from src — does NOT import app.py / Flask)
# ---------------------------------------------------------------------------
from src import (
    load_documents,
    text_split,
    load_embeddings,
    PineconeStore,
    HybridRetriever,
    RAGPipeline,
)

# ---------------------------------------------------------------------------
# RAGAS imports (0.4.x API — Collections metrics)
# ---------------------------------------------------------------------------
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from openai import AsyncOpenAI as _AsyncOpenAI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_DATASET_PATH = "data/test_dataset.json"
RAW_RESULTS_PATH  = "data/eval_rag_results_raw.json"
MEDICAL_CATEGORIES = {"factual_medical", "dietary_management", "symptoms_diagnosis"}
CONV_CATEGORY      = "conversational"

METRIC_LABELS = {
    "context_precision_with_reference": "Context Precision",
    "context_recall":                   "Context Recall",
    "faithfulness":                     "Faithfulness",
    "answer_relevancy":                 "Answer Relevancy",
}


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------

def initialize_rag_pipeline(verbose: bool = False) -> RAGPipeline:
    """
    Build the full RAG pipeline — embeddings → Pinecone → BM25 chunks → hybrid retriever.
    Mirrors the initialisation in CKDChatbotCore without importing Flask.
    """
    print("  Loading embeddings...")
    embeddings = load_embeddings(verbose=verbose)

    print("  Connecting to Pinecone...")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        sys.exit("ERROR: PINECONE_API_KEY not set in environment.")

    vectorstore = PineconeStore(
        api_key=pinecone_api_key,
        index_name="ckd-chatbot",
        dimension=384,
        verbose=verbose,
    )
    vectorstore.init_index()

    print("  Loading document chunks for BM25 corpus...")
    raw_docs  = load_documents(data_dir="data/", verbose=verbose)
    doc_chunks = text_split(raw_docs)
    print(f"  BM25 corpus: {len(doc_chunks)} chunks")

    hybrid_retriever = HybridRetriever(
        embeddings=embeddings,
        pinecone_index=vectorstore.index,
        doc_chunks=doc_chunks,
        k=5,
        beta=0.5,
        verbose=False,
    )

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=500,
        verbose=False,
    )
    pipeline.initialize()
    print("  RAG pipeline ready.\n")
    return pipeline


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def run_queries(pipeline: RAGPipeline, dataset: list, top_k: int = 5) -> list:
    """
    Run every question through the pipeline, collecting full context text
    (required by RAGAS — not just the 100-char preview stored in sources).
    """
    results = []
    total   = len(dataset)

    for i, item in enumerate(dataset, 1):
        question    = item["question"]
        ground_truth = item["ground_truth"]

        print(f"  [{i:>2}/{total}] ({item['category']}) {question[:70]}...")

        result = pipeline.query(question)

        # Strip source tags and leading artefacts
        raw_answer = result.get("answer", "")
        answer = (
            raw_answer
            .replace("[SOURCES_USED]", "")
            .replace("[NO_SOURCES]",   "")
            .strip()
            .lstrip("?!.,;:- \n")
            .strip()
        )

        # Full context strings from LangChain Documents
        contexts = [doc.page_content for doc in result.get("context", [])]

        results.append({
            "id":           item.get("id"),
            "category":     item["category"],
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        })

    return results


# ---------------------------------------------------------------------------
# RAGAS scoring helpers (Collections API — batch_score)
# ---------------------------------------------------------------------------

def score_batch(metric, results: list) -> list:
    """Score each result with the metric; auto-maps fields to what each metric needs."""
    import inspect
    sig     = inspect.signature(metric.ascore)
    allowed = set(sig.parameters.keys()) - {"self"}

    all_fields = {
        "user_input":         lambda r: r["question"],
        "retrieved_contexts": lambda r: r["contexts"],
        "response":           lambda r: r["answer"],
        "reference":          lambda r: r["ground_truth"],
    }

    scores = []
    for i, r in enumerate(results, 1):
        kwargs = {k: fn(r) for k, fn in all_fields.items() if k in allowed}
        try:
            result = metric.score(**kwargs)
            scores.append(result.value if result.value is not None else 0.0)
        except Exception as e:
            print(f"    [WARN] score failed for sample {i}: {type(e).__name__} — using 0.0")
            scores.append(0.0)
    return scores


def print_report(scores_by_metric: dict, label: str):
    print(f"\n{'-'*56}")
    print(f"  {label}")
    print(f"{'-'*56}")
    for col, scores in scores_by_metric.items():
        friendly = METRIC_LABELS.get(col, col)
        valid    = [s for s in scores if s is not None]
        val      = sum(valid) / len(valid) if valid else 0.0
        bar      = "#" * int(val * 20)
        print(f"  {friendly:<30}  {val:.3f}  {bar}")
    print(f"{'-'*56}")


def print_weakest(questions: list, scores: list, metric_name: str, n: int = 3):
    friendly = METRIC_LABELS.get(metric_name, metric_name)
    paired   = sorted(zip(scores, questions), key=lambda x: x[0])
    print(f"\n  Weakest {n} on {friendly}:")
    for score, q in paired[:n]:
        print(f"    [{score:.3f}] {q[:65]}...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CKD Chatbot RAGAS evaluation")
    p.add_argument("--sample",          type=int,   default=None,
                   help="Limit to first N questions")
    p.add_argument("--category",        type=str,   default=None,
                   help="Only run questions from this category")
    p.add_argument("--top-k",           type=int,   default=5,
                   help="Chunks to retrieve per query (default: 5)")
    p.add_argument("--evaluator-model", type=str,   default="gpt-4o-mini",
                   help="OpenAI model used as RAGAS judge")
    p.add_argument("--verbose",         action="store_true",
                   help="Show detailed pipeline output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("\n" + "=" * 56)
    print("  CKD Chatbot — RAGAS Evaluation")
    print("=" * 56)

    # ── Load test dataset ─────────────────────────────────────────────
    if not os.path.exists(TEST_DATASET_PATH):
        sys.exit(
            f"ERROR: {TEST_DATASET_PATH} not found.\n"
            "Copy data/test_dataset.json from KidneyCareAI or run the setup script."
        )

    with open(TEST_DATASET_PATH) as f:
        dataset = json.load(f)

    if args.category:
        dataset = [d for d in dataset if d["category"] == args.category]
        print(f"Category filter: '{args.category}' → {len(dataset)} questions")
    if args.sample:
        dataset = dataset[: args.sample]
        print(f"Sample limit   : {args.sample} questions")

    print(f"Total questions: {len(dataset)}")

    # ── Initialise pipeline ───────────────────────────────────────────
    print("\nInitialising RAG pipeline (this may take ~30s on first run)...")
    pipeline = initialize_rag_pipeline(verbose=args.verbose)

    # ── Run queries ───────────────────────────────────────────────────
    print("Running queries...")
    results = run_queries(pipeline, dataset, top_k=args.top_k)

    # Save raw results (without full contexts to keep file readable)
    os.makedirs("data", exist_ok=True)
    with open(RAW_RESULTS_PATH, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "contexts"} for r in results],
            f, indent=2
        )
    print(f"\nRaw results saved → {RAW_RESULTS_PATH}")

    # ── RAGAS evaluator LLM + embeddings (AsyncOpenAI required) ──────
    import httpx
    _async_client  = _AsyncOpenAI(http_client=httpx.AsyncClient(verify=False))
    evaluator_llm  = llm_factory(args.evaluator_model, client=_async_client, max_tokens=4096)
    evaluator_emb  = RagasOpenAIEmbeddings(client=_async_client)

    ctx_prec   = ContextPrecisionWithReference(llm=evaluator_llm)
    ctx_recall = ContextRecall(llm=evaluator_llm)
    faithful   = Faithfulness(llm=evaluator_llm)
    relevancy  = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb)

    # ── Medical questions — all 4 metrics ────────────────────────────
    medical = [r for r in results if r["category"] in MEDICAL_CATEGORIES]
    if medical:
        print(f"\nEvaluating {len(medical)} medical questions (4 metrics)...")
        med_scores = {
            "context_precision_with_reference": score_batch(ctx_prec,   medical),
            "context_recall":                   score_batch(ctx_recall,  medical),
            "faithfulness":                     score_batch(faithful,    medical),
            "answer_relevancy":                 score_batch(relevancy,   medical),
        }
        print_report(med_scores, f"Medical Questions  (n={len(medical)})")
        med_qs = [r["question"] for r in medical]
        print_weakest(med_qs, med_scores["faithfulness"],                     "faithfulness")
        print_weakest(med_qs, med_scores["context_precision_with_reference"], "context_precision_with_reference")

    # ── Conversational questions — answer relevancy only ─────────────
    conv = [r for r in results if r["category"] == CONV_CATEGORY]
    if conv:
        print(f"\nEvaluating {len(conv)} conversational questions (answer relevancy)...")
        conv_scores = {
            "answer_relevancy": score_batch(relevancy, conv),
        }
        print_report(conv_scores, f"Conversational Questions  (n={len(conv)})")

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
