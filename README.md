# HalluDetect

**LLM Hallucination Detection & Faithfulness Scoring System**



---

## What is HalluDetect?

HalluDetect automatically detects when an LLM "hallucinates" — generating confident but factually incorrect or unsupported content — and assigns a calibrated faithfulness score to its outputs.

The system introduces **HalluScore**, a novel composite metric that combines three independent detection signals into one score, along with a **RAG pipeline** that retrieves reference context automatically so you don't need to supply it manually.

```
Your question  +  LLM-generated answer
                        │
              ┌─────────▼─────────┐
              │   RAG Retriever   │  ← FAISS vector search over knowledge base
              │  (auto-context)   │     retrieves top-k relevant chunks
              └─────────┬─────────┘
                        │  retrieved context
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    NLI Scorer   Semantic Scorer  LLM Judge
   (DeBERTa-v3)  (MiniLM-L6-v2)  (Gemini 2.5)
   entailment    BERTScore F1     reasoning verdict
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                   HalluScore
          0.35·NLI + 0.20·Sem + 0.30·LLM + 0.15·Factuality
                        │
            ┌───────────┼───────────┐
         FAITHFUL   UNCERTAIN  HALLUCINATED
         (≥ 0.55)  (0.45–0.55)  (≤ 0.45)
          + 95% bootstrap confidence interval
```

---

## Features

- **Multi-signal scoring** — NLI entailment + semantic similarity + LLM reasoning combined
- **RAG mode** — automatic context retrieval from a FAISS knowledge base (no manual context needed)
- **Sentence-level granularity** — identifies exactly which sentences are hallucinated
- **Calibrated confidence intervals** — bootstrap 95% CI on every score
- **Web dashboard** — live scoring, weight tuning, RAG KB management
- **REST API** — 7 endpoints for integration into any pipeline
- **15-sample benchmark** — across 5 domains, 3 difficulty levels, 4 hallucination types

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install faiss-cpu sentencepiece
```

> **macOS Python 3.13 users** — `sentencepiece` is required for the DeBERTa-v3 tokenizer. Fixes for semaphore warnings and tokenizer parallelism are already built into the code.

### 2. Download NLTK data (one-time)

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Set your Gemini API key

```bash
cp .env.example .env
# Edit .env and set:  GEMINI_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

> **Free tier:** `gemini-2.5-flash` 

### 4. Start the web dashboard

```bash
cd src
python3 app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## Project Structure

```
hallucination_detector/
│
├── src/
│   ├── config.py            ← all settings: models, weights, API key, RAG params
│   ├── nli_scorer.py        ← DeBERTa-v3 NLI entailment scorer
│   ├── semantic_scorer.py   ← Sentence-Transformer BERTScore scorer
│   ├── llm_judge.py         ← Gemini 1.5 Flash LLM-as-judge
│   ├── hallu_score.py       ← HalluScore composite metric + bootstrap CI
│   ├── benchmark.py         ← 15-sample curated benchmark dataset
│   ├── evaluator.py         ← main benchmark pipeline (Accuracy, F1, ROC-AUC)
│   ├── rag_retriever.py     ← FAISS vector store: ingest, chunk, embed, retrieve
│   ├── rag_evaluator.py     ← RAG-augmented evaluator (auto-fetches context)
│   ├── analysis.py          ← 6 publication-quality plots + evaluation report
│   └── app.py               ← Flask REST API server
│
├── static/
│   └── index.html           ← interactive web dashboard (4 tabs)
│
├── notebooks/
│   └── hallucination_analysis.ipynb
│
├── data/                    ← FAISS index stored here (auto-created)
├── results/                 ← benchmark outputs: JSON, CSV, plots, report
├── requirements.txt
├── .env.example
└── setup_and_run.py
```

---

## HalluScore — The Metric

```
HalluScore(C, G) = 0.35 × NLI(C, G)
                 + 0.20 × Semantic(C, G)
                 + 0.30 × LLMJudge(C, G)
                 + 0.15 × Factuality(C, G)
```

| Component | Model | What it detects |
|-----------|-------|----------------|
| **NLI Score** | `cross-encoder/nli-deberta-v3-base` | Logical contradiction / entailment per sentence |
| **Semantic Score** | `all-MiniLM-L6-v2` | Topic drift, coverage gaps (BERTScore F1) |
| **LLM Judge** | `Gemini 2.5 Flash` | Fabricated facts, subtle reasoning errors |
| **Factuality Bonus** | Rule-based from NLI breakdown | Rewards entailment rate, penalises contradiction rate |

### Verdict thresholds

| Score | Verdict | Meaning |
|-------|---------|---------|
| ≥ 0.55 | **FAITHFUL** | Supported by reference context |
| 0.45 – 0.55 | **UNCERTAIN** | Mixed signals — recommend manual review |
| ≤ 0.45 | **HALLUCINATED** | Contradicts or unsupported by context |

### Comparison with existing metrics

| Metric | Multi-Signal | LLM Reasoning | Single-Pass | Calibrated CI | Open Source |
|--------|:-----------:|:------------:|:----------:|:------------:|:-----------:|
| **HalluScore (ours)** | ✓ | ✓ | ✓ | ✓ | ✓ |
| HHEM (Vectara) | ✗ | ✗ | ✓ | ✗ | ✓ |
| SelfCheckGPT | ✗ | ✗ | ✗ | ✗ | ✓ |
| G-Eval | ✗ | ✓ | ✓ | ✗ | ✗ |
| FActScore | ✗ | ✗ | ✓ | ✗ | Partial |
| AlignScore | Partial | ✗ | ✓ | ✗ | ✓ |

---

## RAG Mode

RAG mode removes the requirement to manually supply a reference context. The system automatically retrieves the most relevant facts from the knowledge base and uses them as context for scoring.

### How retrieval works

```
Query embedded → FAISS cosine search (32 chunks, 12 documents)
→ top-3 chunks retrieved
→ chunks below 0.30 similarity score dropped
→ remaining chunks joined as context string (with source attribution)
→ context passed to NLI + Semantic + LLM pipeline
→ HalluScore returned with sources cited
```

### Built-in knowledge base (12 documents, ~32 chunks)

| Document | Domain |
|----------|--------|
| Photosynthesis | Science |
| Speed of light & special relativity | Science |
| Transformer architecture (Vaswani et al. 2017) | Technology |
| Python programming language | Technology |
| French Revolution | History |
| Diabetes mellitus (Type 1 & 2) | Medicine |
| Aspirin mechanism of action | Medicine |
| Mount Everest | Geography |
| Amazon River | Geography |
| LLMs and hallucination | Technology |
| Retrieval-Augmented Generation | Technology |
| NLI and textual entailment | Technology |

### Add your own documents

```bash
# Via API
curl -X POST http://localhost:5000/api/rag/add \
  -H "Content-Type: application/json" \
  -d '{"title":"Newton Laws","content":"...","domain":"science","source":"textbook"}'

# Via CLI
cd src
python3 rag_evaluator.py --stats
python3 rag_evaluator.py --search "photosynthesis"
python3 rag_evaluator.py \
  --api-key YOUR_KEY \
  --question "How does aspirin work?" \
  --answer "Aspirin blocks COX enzymes to reduce prostaglandins."
```

---

## Running the Benchmark

```bash
cd src

# Full 15-sample benchmark (~5 min due to rate-limit delays between samples)
python3 evaluator.py --api-key YOUR_KEY

# Faster test
python3 evaluator.py --api-key YOUR_KEY --max-samples 5

# Skip Gemini (NLI + Semantic only, no API calls)
python3 evaluator.py --api-key YOUR_KEY --skip-llm

# Generate all analysis plots + report
python3 analysis.py
```

**Outputs saved to `results/`:**

| File | Description |
|------|-------------|
| `benchmark_results.json` | Full per-sample results |
| `benchmark_results.csv` | Summary table |
| `evaluation_report.txt` | Metrics report |
| `score_distributions.png` | Score histogram: faithful vs hallucinated |
| `domain_accuracy.png` | Accuracy per domain |
| `score_correlation.png` | Inter-scorer correlation heatmap |
| `confusion_matrix.png` | Prediction vs ground truth |
| `hallucination_types.png` | Detection rate by hallucination type |
| `metrics_radar.png` | Accuracy / Precision / Recall / F1 / ROC-AUC |

---

## API Reference

All endpoints available at `http://localhost:5000`.

### Manual context scoring

```
POST /api/score
Body: { "context": "...", "generated_text": "..." }
```

### RAG auto-context scoring

```
POST /api/rag/score
Body: {
  "generated_text": "...",     ← required
  "query":          "...",     ← optional (defaults to generated_text)
  "domain":         "science"  ← optional domain filter
}
```

### Search knowledge base

```
POST /api/rag/search
Body: { "query": "...", "top_k": 5 }
```

### Add document to knowledge base

```
POST /api/rag/add
Body: { "title": "...", "content": "...", "domain": "science", "source": "custom" }
```

### Other endpoints

```
GET /api/rag/stats     → KB statistics (documents, chunks, domains)
GET /api/benchmark     → cached benchmark results
GET /api/health        → server status
```

### Example response (`/api/rag/score`)

```json
{
  "hallu_score": 0.3412,
  "verdict": "HALLUCINATED",
  "confidence": 0.8234,
  "is_hallucinated": true,
  "scores": {
    "nli": 0.2100,
    "semantic": 0.4800,
    "llm_judge": 0.3200
  },
  "details": {
    "hallucination_rate": 0.6667,
    "hallucinated_sentences": 2,
    "total_sentences": 3,
    "bertscore_f1": 0.5123
  },
  "sentence_analysis": [
    { "sentence": "Aspirin blocks serotonin receptors...",
      "label": "CONTRADICTION", "nli_score": 0.0821 },
    { "sentence": "It stimulates endorphin production...",
      "label": "CONTRADICTION", "nli_score": 0.1203 },
    { "sentence": "Aspirin is used for pain relief.",
      "label": "ENTAILMENT", "nli_score": 0.9102 }
  ],
  "rag": {
    "sources": [
      { "rank": 1, "title": "Aspirin mechanism of action",
        "domain": "medicine", "score": 0.8823 }
    ],
    "context_preview": "[Source: Aspirin mechanism of action] Aspirin works by...",
    "query_used": "How does aspirin work?"
  }
}
```

---

## Web Dashboard

The dashboard at `http://localhost:5000` has 4 tabs:

| Tab | What you can do |
|-----|----------------|
| **Live Scorer** | Paste context + LLM output, run scoring, see sentence-level NLI breakdown |
| **HalluScore Metric** | Adjust NLI/Semantic/LLM/Factuality weights with sliders, view benchmark metrics |
| **Architecture** | System comparison vs HHEM, SelfCheckGPT, G-Eval, FActScore |
| **RAG Mode** | Score without manual context, search KB, add documents, view retrieved sources |

---

## Configuration

All settings in `src/config.py`:

```python
GEMINI_MODEL      = "gemini-1.5-flash"           # 1,500 req/day free tier
NLI_MODEL_NAME    = "cross-encoder/nli-deberta-v3-base"
SENTENCE_MODEL    = "all-MiniLM-L6-v2"

# HalluScore weights (must sum to 1.0)
WEIGHT_NLI        = 0.35
WEIGHT_SEMANTIC   = 0.20
WEIGHT_LLM_JUDGE  = 0.30
WEIGHT_FACTUALITY = 0.15

# RAG
RAG_CHUNK_SIZE    = 80          # words per chunk
RAG_CHUNK_OVERLAP = 20          # overlap between chunks
RAG_TOP_K         = 3           # chunks retrieved per query
```

---

## Troubleshooting

### `ImportError: DebertaV2Tokenizer requires SentencePiece`

```bash
pip install sentencepiece
```

### macOS Python 3.13 — semaphore warnings

Already fixed in the codebase via:
- `TOKENIZERS_PARALLELISM=false` set before HuggingFace imports in every scorer file
- `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` in `app.py`
- `multiprocessing.set_start_method("fork", force=True)` for macOS in `app.py`

If the warning still appears, it is cosmetic only — the app continues running correctly.

### Gemini 429 rate limit errors

The benchmark auto-waits 15 seconds between samples. If you exceed the daily quota:
- Use `--skip-llm` to run NLI + Semantic scoring only (no API calls)
- Create a new API key at [aistudio.google.com](https://aistudio.google.com)
- Switch from `gemini-2.5-flash` (20/day) to `gemini-1.5-flash` (1,500/day) in `config.py`

### RAG index stale after changing chunk size

```bash
rm data/rag_index*.pkl
# Restart — index rebuilds automatically with versioned filename
```

### `NoneType has no attribute strip` — RAG endpoint error

Already fixed. Caused by frontend sending `null` for optional fields. The fix wraps all `.get()` calls with `or ""` to handle null values safely.

---

## Gemini API Free Tier Limits

| Model | Requests/day | Requests/min |
|-------|:-----------:|:-----------:|
| `gemini-1.5-flash` ✓ recommended | 1,500 | 15 |
| `gemini-2.5-flash` | 20 | 5 |

---

## Sample Test Inputs
 
### Live Scorer tab
 
**Context:**
```
Aspirin works by irreversibly inhibiting COX-1 and COX-2 enzymes, reducing
prostaglandin production. At low doses it prevents platelet aggregation,
reducing the risk of blood clots.
```
 
**Faithful answer:**
```
Aspirin inhibits COX-1 and COX-2 enzymes, which reduces the production of
prostaglandins and relieves pain and inflammation. At low doses, it also
prevents platelet aggregation, lowering the risk of blood clots forming.
```
→ Expected: **FAITHFUL**
 
**Hallucinated answer:**
```
Aspirin works by blocking serotonin receptors in the brain and stimulates
endorphin production. To prevent clots, it activates platelets and increases
their aggregation rate at high doses.
```
→ Expected: **HALLUCINATED**
 
### RAG Mode tab
 
**Question:** `How does aspirin work to relieve pain and prevent blood clots?`
 
**Faithful answer:**
```
Aspirin inhibits COX-1 and COX-2 enzymes, reducing prostaglandin production
and relieving pain. At low doses it prevents platelets from clumping together,
reducing the risk of heart attacks and strokes.
```
→ Expected: **FAITHFUL** — RAG retrieves aspirin KB article automatically
 
**Hallucinated answer:**
```
Aspirin works by blocking serotonin receptors in the brain. It also stimulates
endorphin production, which explains its anti-inflammatory effects.
```
→ Expected: **HALLUCINATED**

---

## Novel Contributions

1. **HalluScore** — first open-source metric combining NLI + semantic + LLM signals in a single pass with bootstrap confidence intervals
2. **Zero-context RAG detection** — retrieval quality (cosine similarity ≥ 0.30 threshold) integrated into the scoring pipeline
3. **Hallucination type taxonomy** — per-type detection rates expose which types current metrics struggle with most
4. **Calibrated uncertainty** — 95% CI surfaces ambiguous cases for human review instead of forcing a binary verdict

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| NLI model | `cross-encoder/nli-deberta-v3-base` via HuggingFace Transformers |
| Embeddings | `all-MiniLM-L6-v2` via Sentence-Transformers |
| Vector search | FAISS (`faiss-cpu`) |
| LLM judge | Gemini 2.5 Flash via `google-genai` SDK |
| Backend | Flask + Flask-CORS |
| Analysis | matplotlib, seaborn, scipy, scikit-learn |
| Frontend | Vanilla HTML/CSS/JS (dark theme, single file) |

---
