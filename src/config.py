# config.py — Central configuration for the Hallucination Detection System

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# ─── Model Config ────────────────────────────────────────────────────────────
GEMINI_MODEL         = "gemini-2.5-flash"          # fast + cheap for bulk eval
GEMINI_MODEL_PRO     = "gemini-2.5-pro"            # higher quality responses

# NLI backbone for entailment scoring
NLI_MODEL_NAME       = "cross-encoder/nli-deberta-v3-base"  # best lightweight NLI
SENTENCE_MODEL_NAME  = "all-MiniLM-L6-v2"                  # semantic similarity

# ─── Scoring Thresholds ──────────────────────────────────────────────────────
FAITHFULNESS_THRESHOLD      = 0.5   # below this → hallucination flagged
HIGH_CONFIDENCE_THRESHOLD   = 0.75
LOW_CONFIDENCE_THRESHOLD    = 0.25

# ─── Benchmark Datasets ──────────────────────────────────────────────────────
# We build a custom benchmark + use TruthfulQA samples
CUSTOM_BENCHMARK_PATH = "data/custom_benchmark.json"
RESULTS_DIR           = "results/"

# ─── Scoring Weights (for composite HalluScore) ──────────────────────────────
# HalluScore = w1*NLI + w2*SemanticSim + w3*LLMJudge + w4*Factuality
WEIGHT_NLI         = 0.35
WEIGHT_SEMANTIC    = 0.20
WEIGHT_LLM_JUDGE   = 0.30
WEIGHT_FACTUALITY  = 0.15

# ─── System Prompt for LLM Judge ─────────────────────────────────────────────
LLM_JUDGE_SYSTEM_PROMPT = """You are an expert fact-checker and hallucination detector.
Given a CONTEXT (ground truth) and a CLAIM (LLM-generated statement), analyze whether
the claim is faithful to the context or hallucinated.

Respond ONLY with a JSON object in this exact format:
{
  "verdict": "FAITHFUL" | "HALLUCINATED" | "UNCERTAIN",
  "confidence": <float 0.0-1.0>,
  "reason": "<one sentence explanation>",
  "claim_type": "factual" | "inferential" | "contradictory" | "fabricated"
}
"""

# ─── RAG Configuration ────────────────────────────────────────────────────────
RAG_EMBED_MODEL   = "all-MiniLM-L6-v2"   # same model as semantic scorer
RAG_CHUNK_SIZE    = 200                   # words per chunk
RAG_CHUNK_OVERLAP = 40                    # word overlap between chunks
RAG_TOP_K         = 3                     # chunks to retrieve per query
RAG_INDEX_PATH    = "data/rag_index"      # where FAISS index is saved
