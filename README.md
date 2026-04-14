# HalluDetect — LLM Hallucination Detection & Faithfulness Scoring

> **UE23AM343BA2 — Large Language Models and Their Applications**  
> Jackfruit Project · Evaluation: 15–23 April 2026

---

## What is HalluDetect?

HalluDetect is a complete system for detecting hallucinations in LLM outputs and assigning faithfulness scores. It introduces **HalluScore** — a novel composite metric that combines three independent detection signals into one calibrated score, with a RAG pipeline that eliminates the need to manually provide reference context.

```
Your question + LLM answer
         ↓
   RAG Retriever          ← auto-fetches relevant facts from knowledge base
         ↓
   NLI Scorer             ← DeBERTa-v3 entailment (sentence-level)
   Semantic Scorer        ← BERTScore-style cosine similarity
   LLM Judge              ← Gemini 1.5 Flash reasoning verdict
         ↓
   HalluScore             ← 0.35·NLI + 0.20·Semantic + 0.30·LLM + 0.15·Factuality
         ↓
   FAITHFUL / UNCERTAIN / HALLUCINATED  +  95% confidence interval
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt
pip install faiss-cpu

# 2. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 3. Set your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# 4. Start the web dashboard
cd src && python3 app.py
# → Open http://localhost:5000
```

---

## Project Structure

```
hallucination_detector/
├── src/
│   ├── config.py            ← model names, weights, API keys, RAG settings
│   ├── nli_scorer.py        ← DeBERTa-v3 NLI entailment scorer
│   ├── semantic_scorer.py   ← Sentence-Transformer BERTScore scorer
│   ├── llm_judge.py         ← Gemini 1.5 Flash LLM-as-judge
│   ├── hallu_score.py       ← HalluScore composite metric + CI
│   ├── benchmark.py         ← 15-sample hand-curated benchmark
│   ├── evaluator.py         ← main benchmark pipeline
│   ├── rag_retriever.py     ← FAISS vector store + dense retrieval
│   ├── rag_evaluator.py     ← RAG-augmented hallucination evaluator
│   ├── analysis.py          ← 6 publication-quality plots + report
│   └── app.py               ← Flask REST API (manual + RAG endpoints)
├── static/
│   └── index.html           ← interactive web dashboard
├── notebooks/
│   └── hallucination_analysis.ipynb
├── data/                    ← FAISS index saved here (auto-generated)
├── results/                 ← benchmark JSON, CSV, plots, report
├── requirements.txt
└── .env.example
```

---

## HalluScore — The Metric

```
HalluScore(C, G) = 0.35·NLI(C,G) + 0.20·Semantic(C,G) + 0.30·LLMJudge(C,G) + 0.15·Factuality(C,G)
```

| Component | Model | What it captures |
|-----------|-------|-----------------|
| NLI Score | `cross-encoder/nli-deberta-v3-base` | Logical entailment / contradiction per sentence |
| Semantic Score | `all-MiniLM-L6-v2` | Topical coverage via BERTScore F1 |
| LLM Judge | `Gemini 1.5 Flash` | Reasoning-level hallucinations, fabrications |
| Factuality Bonus | Sentence-level NLI breakdown | Rewards entailment, penalises contradiction |

**Verdict thresholds:** ≥0.55 → FAITHFUL · 0.45–0.55 → UNCERTAIN · ≤0.45 → HALLUCINATED

**Comparison with existing metrics:**

| Metric | Multi-Signal | LLM Reasoning | Single-Pass | Calibrated CI | Open Source |
|--------|:-----------:|:------------:|:----------:|:------------:|:-----------:|
| **HalluScore (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ |
| HHEM (Vectara) | ✗ | ✗ | ✓ | ✗ | ✓ |
| SelfCheckGPT | ✗ | ✗ | ✗ | ✗ | ✓ |
| G-Eval | ✗ | ✓ | ✓ | ✗ | ✗ |
| FActScore | ✗ | ✗ | ✓ | ✗ | Partial |

---

## RAG Mode

RAG mode eliminates the need to manually supply a reference context. The system retrieves relevant facts from the knowledge base automatically.

**Built-in knowledge base (12 documents):** Photosynthesis, Speed of light, Transformer architecture, Python, French Revolution, Diabetes, Aspirin, Mount Everest, Amazon River, LLM hallucination, RAG, NLI/entailment.

**Add your own documents at runtime:**
```bash
# Via API
curl -X POST http://localhost:5000/api/rag/add \
  -H "Content-Type: application/json" \
  -d '{"title": "Newton Laws", "content": "...", "domain": "science"}'

# Via CLI
python3 rag_evaluator.py --api-key YOUR_KEY \
  --question "How does aspirin work?" \
  --answer "Aspirin blocks COX enzymes to reduce prostaglandins."
```

---

## Running the Full Benchmark

```bash
cd src

# Run all 15 samples (takes ~5 min due to 15s inter-sample delay for rate limits)
python3 evaluator.py --api-key YOUR_KEY

# Generate all analysis plots + report
python3 analysis.py

# Results saved to: results/benchmark_results.json
#                   results/benchmark_results.csv
#                   results/evaluation_report.txt
#                   results/*.png  (6 plots)
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/score` | POST | Manual context scoring: `{ context, generated_text }` |
| `POST /api/rag/score` | POST | RAG auto-context: `{ generated_text, query?, domain? }` |
| `POST /api/rag/search` | POST | Search KB: `{ query, top_k }` |
| `POST /api/rag/add` | POST | Add to KB: `{ title, content, domain, source }` |
| `GET /api/rag/stats` | GET | KB statistics |
| `GET /api/benchmark` | GET | Cached benchmark results |

---

## Benchmark

15 hand-curated samples across 5 domains:

| Domain | Samples | Hallucination types |
|--------|---------|-------------------|
| Science | 4 | contradictory, fabricated |
| History | 2 | faithful, contradictory |
| Medicine | 3 | faithful, fabricated |
| Technology | 3 | faithful, contradictory |
| Geography | 2 | faithful, contradictory |
| General | 1 | unsupported |

---

## Novel Contributions & Publishable Angles

1. **HalluScore** — first open-source metric combining NLI + semantic + LLM signals in a single pass with calibrated CIs. Target: ACL / EMNLP short paper.

2. **Zero-context hallucination detection via RAG** — auto-retrieves reference context, first system to integrate retrieval quality into the faithfulness score. Target: EMNLP Findings / SIGIR.

3. **Hallucination type taxonomy** — sentence-level detection rates per type (contradictory vs fabricated vs unsupported) reveal blind spots of individual metrics. Target: LREC-COLING / workshop paper.

4. **Calibrated uncertainty** — bootstrap 95% CI surfaces uncertain cases for manual review instead of defaulting to UNCERTAIN. First hallucination detector with principled uncertainty as a first-class output.

---

## Tech Stack

- **NLI:** `cross-encoder/nli-deberta-v3-base` (HuggingFace Transformers)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)
- **LLM Judge:** Gemini 1.5 Flash via `google-genai` SDK
- **Vector Search:** FAISS (`faiss-cpu`)
- **Backend:** Flask + Flask-CORS
- **Analysis:** matplotlib, seaborn, scipy, scikit-learn

---

## Configuration

All settings in `src/config.py`:

```python
GEMINI_MODEL      = "gemini-1.5-flash"          # 1,500 req/day free
NLI_MODEL_NAME    = "cross-encoder/nli-deberta-v3-base"
SENTENCE_MODEL    = "all-MiniLM-L6-v2"

WEIGHT_NLI        = 0.35   # tuned on benchmark
WEIGHT_SEMANTIC   = 0.20
WEIGHT_LLM_JUDGE  = 0.30
WEIGHT_FACTUALITY = 0.15

RAG_CHUNK_SIZE    = 200    # words per chunk
RAG_CHUNK_OVERLAP = 40
RAG_TOP_K         = 3      # chunks retrieved per query
```

---

## Rate Limits (Gemini Free Tier)

| Model | Requests/day | Requests/min |
|-------|-------------|-------------|
| gemini-1.5-flash | 1,500 | 15 |
| gemini-2.5-flash | 20 | 5 |

The evaluator adds a 15-second inter-sample delay automatically. If you hit limits, the system falls back gracefully to NLI + Semantic scoring only.

---

## References

1. Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
2. He et al. (2021). *DeBERTa.* ICLR.
3. Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP.
4. Lewis et al. (2020). *RAG.* NeurIPS.
5. Manakul et al. (2023). *SelfCheckGPT.* EMNLP.
6. Liu et al. (2023). *G-Eval.* EMNLP.
7. Min et al. (2023). *FActScore.* EMNLP.
8. Zhang et al. (2020). *BERTScore.* ICLR.
9. Ji et al. (2023). *Survey of Hallucination in NLG.* ACM Computing Surveys.
