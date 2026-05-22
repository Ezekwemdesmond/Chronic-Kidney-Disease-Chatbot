"""
CKD Chatbot — End-to-End Evaluation Script
==========================================
Tests the complete pipeline: clinical biomarkers → ML prediction → RAG advice.

For each synthetic patient profile the script:
  1. Runs the Random Forest classifier  → CKD likely / unlikely
  2. Queries the RAG pipeline           → personalised advice
  3. Asks GPT-4o-mini to judge whether the advice is coherent with the prediction

Coherence is scored 1–5:
  1 = Contradicts or ignores the prediction
  3 = Partially appropriate
  5 = Well-tailored, specific, and fully aligned

Six synthetic profiles are included:
  - 3 clearly CKD  (high creatinine, anaemia, hypertension, proteinuria)
  - 3 clearly not-CKD  (normal biomarker values)

Usage
-----
  python evaluate_e2e.py              # Full run (6 profiles)
  python evaluate_e2e.py --verbose    # Show RAG pipeline logs

Output
------
  Console : per-profile prediction + coherence score + judge verdict
  data/   : eval_e2e_results.json
"""

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

import argparse
import joblib

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from src import (
    load_documents,
    text_split,
    load_embeddings,
    PineconeStore,
    HybridRetriever,
    RAGPipeline,
    preprocess_input,
)

# ---------------------------------------------------------------------------
# Synthetic patient profiles
# ---------------------------------------------------------------------------
# Each profile is a dict matching TOP_FEATURES keys.
# Categorical values must match what the LabelEncoder was fitted on:
#   hypertension, diabetes_mellitus, peda_edema → 'yes' | 'no'
#   appetite                                    → 'good' | 'poor'
# Albumin and sugar are ordinal integers (0–4 / 0–5).

PROFILES = [
    # ── CKD profiles ──────────────────────────────────────────────────
    {
        "label":        "CKD — Severe (diabetic nephropathy)",
        "expected":     "Kidney disease likely",
        "hypertension": "yes",
        "diabetes_mellitus": "yes",
        "serum_creatinine":  7.2,
        "blood_urea":        145.0,
        "haemoglobin":       8.2,
        "specific_gravity":  1.005,
        "albumin":           4,
        "blood_glucose_random": 280.0,
        "packed_cell_volume":   21.0,
        "red_blood_cell_count": 2.6,
        "sodium":            128.0,
        "blood_pressure":    100.0,
        "peda_edema":        "yes",
        "sugar":             4,
        "appetite":          "poor",
    },
    {
        "label":        "CKD — Moderate (hypertensive nephrosclerosis)",
        "expected":     "Kidney disease likely",
        "hypertension": "yes",
        "diabetes_mellitus": "no",
        "serum_creatinine":  3.8,
        "blood_urea":        72.0,
        "haemoglobin":       10.1,
        "specific_gravity":  1.010,
        "albumin":           2,
        "blood_glucose_random": 130.0,
        "packed_cell_volume":   29.0,
        "red_blood_cell_count": 3.2,
        "sodium":            136.0,
        "blood_pressure":    90.0,
        "peda_edema":        "yes",
        "sugar":             0,
        "appetite":          "poor",
    },
    {
        "label":        "CKD — Mild (Stage 3, early anaemia)",
        "expected":     "Kidney disease likely",
        "hypertension": "yes",
        "diabetes_mellitus": "no",
        "serum_creatinine":  2.1,
        "blood_urea":        48.0,
        "haemoglobin":       11.5,
        "specific_gravity":  1.015,
        "albumin":           1,
        "blood_glucose_random": 115.0,
        "packed_cell_volume":   34.0,
        "red_blood_cell_count": 3.8,
        "sodium":            138.0,
        "blood_pressure":    85.0,
        "peda_edema":        "no",
        "sugar":             0,
        "appetite":          "good",
    },
    # ── Non-CKD profiles ──────────────────────────────────────────────
    {
        "label":        "No CKD — Young healthy adult",
        "expected":     "Kidney disease unlikely",
        "hypertension": "no",
        "diabetes_mellitus": "no",
        "serum_creatinine":  0.8,
        "blood_urea":        14.0,
        "haemoglobin":       15.2,
        "specific_gravity":  1.022,
        "albumin":           0,
        "blood_glucose_random": 92.0,
        "packed_cell_volume":   44.0,
        "red_blood_cell_count": 5.0,
        "sodium":            141.0,
        "blood_pressure":    70.0,
        "peda_edema":        "no",
        "sugar":             0,
        "appetite":          "good",
    },
    {
        "label":        "No CKD — Middle-aged, slightly elevated BP",
        "expected":     "Kidney disease unlikely",
        "hypertension": "no",
        "diabetes_mellitus": "no",
        "serum_creatinine":  1.1,
        "blood_urea":        22.0,
        "haemoglobin":       13.8,
        "specific_gravity":  1.018,
        "albumin":           0,
        "blood_glucose_random": 108.0,
        "packed_cell_volume":   40.0,
        "red_blood_cell_count": 4.6,
        "sodium":            139.0,
        "blood_pressure":    80.0,
        "peda_edema":        "no",
        "sugar":             0,
        "appetite":          "good",
    },
    {
        "label":        "No CKD — Older adult, controlled diabetes",
        "expected":     "Kidney disease unlikely",
        "hypertension": "no",
        "diabetes_mellitus": "yes",
        "serum_creatinine":  1.0,
        "blood_urea":        18.0,
        "haemoglobin":       13.0,
        "specific_gravity":  1.020,
        "albumin":           0,
        "blood_glucose_random": 145.0,
        "packed_cell_volume":   38.0,
        "red_blood_cell_count": 4.3,
        "sodium":            140.0,
        "blood_pressure":    78.0,
        "peda_edema":        "no",
        "sugar":             1,
        "appetite":          "good",
    },
]

# Features that go into preprocess_input (same order as TOP_FEATURES minus label/expected)
FORM_KEYS = [
    "hypertension", "red_blood_cell_count", "specific_gravity", "appetite",
    "blood_glucose_random", "blood_urea", "diabetes_mellitus", "haemoglobin",
    "albumin", "packed_cell_volume", "sodium", "blood_pressure", "peda_edema",
    "serum_creatinine", "sugar",
]

COHERENCE_PROMPT = """\
A CKD risk assessment tool predicted the following for a patient:

Prediction : {prediction}

The AI medical assistant then generated this personalised advice:

---
{advice}
---

Evaluate whether the advice is coherent and appropriate given the prediction.

Score the coherence 1–5:
  5 = Excellent — advice is specific, urgent level matches prediction, no contradictions
  4 = Good — mostly appropriate with minor gaps
  3 = Partial — some relevant content but missing key elements for this prediction
  2 = Poor — largely misaligned or generic despite clear prediction
  1 = Bad — contradicts prediction or completely off-topic

Respond ONLY with valid JSON (no markdown fences):
{{
  "score": <int 1-5>,
  "verdict": "<one sentence summary>",
  "issues": "<specific problems, or empty string if none>"
}}"""


# ---------------------------------------------------------------------------
# Pipeline initialisation helpers
# ---------------------------------------------------------------------------

def load_ml_model():
    model    = joblib.load("./data/kidney_disease_rf_model.pkl")
    encoders = joblib.load("./data/encoders.pkl")
    return model, encoders


def initialize_rag_pipeline(verbose: bool = False) -> RAGPipeline:
    embeddings = load_embeddings(verbose=verbose)

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

    raw_docs   = load_documents(data_dir="data/", verbose=verbose)
    doc_chunks = text_split(raw_docs)

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
    return pipeline


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def run_profile(profile: dict, model, rag_pipeline: RAGPipeline) -> dict:
    """Run the full pipeline for one synthetic patient profile."""
    form_data = {k: profile[k] for k in FORM_KEYS if k in profile}

    # 1. ML prediction
    features   = preprocess_input(form_data)
    prediction = model.predict(features)[0]
    result_str = "Kidney disease likely" if prediction == 1 else "Kidney disease unlikely"

    # 2. RAG advice
    query = (
        "The user is at risk of kidney disease. "
        "What advice can you provide to help them manage this risk?"
        if prediction == 1 else
        "The user is not at immediate risk of kidney disease. "
        "What advice can you provide to help them maintain good kidney health?"
    )
    rag_result = rag_pipeline.query(query)
    raw_advice = rag_result.get("answer", "")
    advice = (
        raw_advice
        .replace("[SOURCES_USED]", "")
        .replace("[NO_SOURCES]",   "")
        .strip()
        .lstrip("?!.,;:- \n")
        .strip()
    )

    return {
        "label":      profile["label"],
        "expected":   profile["expected"],
        "prediction": result_str,
        "correct":    result_str == profile["expected"],
        "advice":     advice,
    }


def check_coherence(client: OpenAI, prediction: str, advice: str) -> dict:
    """Ask GPT-4o-mini to judge whether advice is coherent with the prediction."""
    prompt = COHERENCE_PROMPT.format(prediction=prediction, advice=advice)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=250,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "verdict": "Parse error", "issues": raw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CKD Chatbot end-to-end evaluation")
    p.add_argument("--verbose", action="store_true",
                   help="Show detailed pipeline output")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  CKD Chatbot — End-to-End Evaluation")
    print("=" * 60)

    # ── Initialise components ─────────────────────────────────────────
    print("\nLoading ML model...")
    model, _ = load_ml_model()

    print("Initialising RAG pipeline (this may take ~30s)...")
    rag_pipeline = initialize_rag_pipeline(verbose=args.verbose)

    import httpx as _httpx
    openai_client = OpenAI(http_client=_httpx.Client(verify=False))

    # ── Run each profile ──────────────────────────────────────────────
    all_results = []
    correct_predictions = 0

    print(f"\nRunning {len(PROFILES)} synthetic patient profiles...\n")

    for i, profile in enumerate(PROFILES, 1):
        print(f"  Profile {i}: {profile['label']}")

        result = run_profile(profile, model, rag_pipeline)

        pred_icon = "✓" if result["correct"] else "✗"
        print(f"    Prediction : {result['prediction']}  {pred_icon} (expected: {result['expected']})")

        # Coherence check
        coherence = check_coherence(openai_client, result["prediction"], result["advice"])
        score   = coherence.get("score", 0)
        verdict = coherence.get("verdict", "")
        issues  = coherence.get("issues", "")
        bar     = "█" * score + "░" * (5 - score)
        print(f"    Coherence  : {score}/5  [{bar}]  {verdict}")
        if issues:
            print(f"    Issues     : {issues}")
        print()

        if result["correct"]:
            correct_predictions += 1

        all_results.append({**result, "coherence": coherence})

    # ── Summary ───────────────────────────────────────────────────────
    total    = len(PROFILES)
    accuracy = correct_predictions / total
    avg_coh  = sum(
        r["coherence"].get("score", 0) for r in all_results
    ) / total

    ckd_profiles    = [r for r in all_results if r["expected"] == "Kidney disease likely"]
    non_ckd_profiles = [r for r in all_results if r["expected"] == "Kidney disease unlikely"]
    ckd_coh     = sum(r["coherence"].get("score", 0) for r in ckd_profiles)    / max(len(ckd_profiles), 1)
    non_ckd_coh = sum(r["coherence"].get("score", 0) for r in non_ckd_profiles) / max(len(non_ckd_profiles), 1)

    print(f"{'─'*60}")
    print(f"  E2E Summary")
    print(f"{'─'*60}")
    print(f"  ML Prediction Accuracy      {correct_predictions}/{total}  ({accuracy:.0%})")
    print(f"  Avg Coherence Score         {avg_coh:.2f} / 5")
    print(f"    └─ CKD profiles           {ckd_coh:.2f} / 5")
    print(f"    └─ Non-CKD profiles       {non_ckd_coh:.2f} / 5")
    print(f"{'─'*60}")

    # Flag misclassifications
    wrong = [r for r in all_results if not r["correct"]]
    if wrong:
        print(f"\n  ⚠  Misclassified ({len(wrong)}):")
        for r in wrong:
            print(f"     • {r['label']}")
            print(f"       Got: {r['prediction']}  |  Expected: {r['expected']}")

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    with open("data/eval_e2e_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → data/eval_e2e_results.json")
    print("Evaluation complete.\n")


if __name__ == "__main__":
    main()
