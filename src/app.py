# app.py — Flask REST API for the Hallucination Detection System
# Includes original /api/score (manual context) + new RAG endpoints.

import os
import sys
import json
import logging
import traceback
import multiprocessing

# ── macOS Python 3.13 fixes — must happen before ANY other imports ────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # stop HF fast tokenizer spawning threads
os.environ["OMP_NUM_THREADS"]        = "1"        # stop OpenMP spawning threads
os.environ["MKL_NUM_THREADS"]        = "1"        # stop MKL spawning threads

# Force fork start method on macOS to prevent semaphore leaks from spawn/forkserver
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass  # already set — fine

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_evaluator     = None
_rag_evaluator = None


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        from config import GEMINI_API_KEY, GEMINI_MODEL
        from evaluator import HallucinationEvaluator
        api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        logger.info(f"Starting evaluator with model: {GEMINI_MODEL}")
        _evaluator = HallucinationEvaluator(
            api_key=api_key, skip_nli=False, skip_llm_judge=False, use_cot=False
        )
    return _evaluator


def get_rag_evaluator():
    global _rag_evaluator
    if _rag_evaluator is None:
        from config import GEMINI_API_KEY
        from rag_evaluator import RAGHallucinationEvaluator
        api_key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        logger.info("Initialising RAG evaluator...")
        _rag_evaluator = RAGHallucinationEvaluator(
            api_key=api_key, top_k=3, skip_llm_judge=False
        )
    return _rag_evaluator


def _build_score_response(result: dict) -> dict:
    """Shared helper: convert evaluator result → clean API response dict."""
    summary = result.get("summary", {})
    nli     = result.get("nli", {})
    sem     = result.get("semantic", {})
    llm     = result.get("llm_judge", {})

    response = {
        "hallu_score":     round(summary.get("hallu_score", 0.5), 4),
        "verdict":         summary.get("verdict", "UNCERTAIN"),
        "confidence":      round(summary.get("confidence", 0.5), 4),
        "is_hallucinated": summary.get("is_hallucinated", False),
        "scores": {
            "nli":       round(summary.get("nli_score", 0.5), 4),
            "semantic":  round(summary.get("semantic_score", 0.5), 4),
            "llm_judge": round(summary.get("llm_judge_score", 0.5), 4),
        },
        "details": {
            "hallucination_rate":     round(summary.get("hallucination_rate", 0), 4),
            "hallucinated_sentences": nli.get("hallucinated_sentences", 0),
            "total_sentences":        nli.get("total_sentences", 0),
            "bertscore_f1":           round(sem.get("bertscore_f1", 0), 4),
            "llm_reason":             (llm.get("sentence_scores") or [{}])[0].get("reason", ""),
        },
        "sentence_analysis": []
    }

    for s in (nli.get("sentence_scores") or [])[:10]:
        response["sentence_analysis"].append({
            "sentence":      s.get("sentence", ""),
            "label":         s.get("predicted_label", "UNKNOWN"),
            "nli_score":     round(s.get("faithfulness_score", 0.5), 4),
            "entailment":    round(s.get("entailment_prob", 0), 4),
            "contradiction": round(s.get("contradiction_prob", 0), 4),
        })

    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("../static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "HalluDetect API running (with RAG)"})


# ── Original manual-context endpoint (unchanged) ──────────────────────────────

@app.route("/api/score", methods=["POST"])
def score():
    """
    Score with a manually supplied reference context.
    Body: { "context": "...", "generated_text": "..." }
    """
    try:
        data           = request.get_json()
        context        = (data.get("context") or "").strip()
        generated_text = (data.get("generated_text") or "").strip()

        if not context or not generated_text:
            return jsonify({"error": "Both 'context' and 'generated_text' are required"}), 400

        result   = get_evaluator().evaluate_single(context, generated_text)
        response = _build_score_response(result)
        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── RAG endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/rag/score", methods=["POST"])
def rag_score():
    """
    RAG-augmented scoring: context is retrieved automatically from the KB.

    Body:
      {
        "generated_text": "...",      ← required: LLM output to check
        "query":          "...",      ← optional: question (defaults to generated_text)
        "domain":         "science"   ← optional: restrict KB search to domain
      }

    Returns: same shape as /api/score + rag_metadata field
    """
    try:
        data           = request.get_json()
        generated_text = (data.get("generated_text") or "").strip()
        query          = (data.get("query") or "").strip() or None
        domain         = (data.get("domain") or "").strip() or None

        if not generated_text:
            return jsonify({"error": "'generated_text' is required"}), 400

        rag    = get_rag_evaluator()
        result = rag.evaluate(
            generated_text = generated_text,
            query          = query,
            domain_filter  = domain
        )

        if "error" in result:
            return jsonify(result), 400

        response = _build_score_response(result)

        # Attach RAG metadata to response
        meta = result.get("rag_metadata", {})
        response["rag"] = {
            "context_preview": meta.get("context_preview", ""),
            "sources":         meta.get("sources_used", []),
            "query_used":      meta.get("query_used", generated_text[:100]),
            "top_k":           meta.get("top_k", 3),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/search", methods=["POST"])
def rag_search():
    """
    Search the knowledge base without scoring.
    Useful for exploring what context would be retrieved for a query.

    Body: { "query": "...", "top_k": 5 }
    """
    try:
        data   = request.get_json()
        query  = (data.get("query") or "").strip()
        top_k  = int(data.get("top_k", 5))

        if not query:
            return jsonify({"error": "'query' is required"}), 400

        rag     = get_rag_evaluator()
        results = rag.search_knowledge_base(query, top_k=top_k)
        return jsonify({"query": query, "results": results})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/add", methods=["POST"])
def rag_add_document():
    """
    Add a new document to the knowledge base at runtime.

    Body:
      {
        "title":   "My Document",
        "content": "Full text of the document...",
        "domain":  "science",     ← optional
        "source":  "custom"       ← optional
      }
    """
    try:
        data    = request.get_json()
        title   = (data.get("title") or "").strip()
        content = (data.get("content") or "").strip()
        domain  = (data.get("domain") or "general").strip()
        source  = (data.get("source") or "custom").strip()

        if not title or not content:
            return jsonify({"error": "'title' and 'content' are required"}), 400

        rag    = get_rag_evaluator()
        doc_id = rag.add_to_knowledge_base(title, content, domain, source)

        return jsonify({
            "success": True,
            "doc_id":  doc_id,
            "message": f"Document '{title}' added and index rebuilt.",
            "kb_stats": rag.get_knowledge_base_stats()
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    """Return knowledge base statistics."""
    try:
        rag = get_rag_evaluator()
        return jsonify(rag.get_knowledge_base_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Existing endpoints ────────────────────────────────────────────────────────

@app.route("/api/benchmark", methods=["GET"])
def get_benchmark_results():
    results_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "benchmark_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No benchmark results found. Run evaluator.py first."}), 404


@app.route("/api/demo", methods=["GET"])
def demo_examples():
    examples = [
        {
            "name": "Faithful Science (manual)",
            "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. It occurs in chloroplasts.",
            "generated": "Plants convert sunlight, water, and CO2 into glucose and oxygen through photosynthesis in their chloroplasts.",
            "expected": "FAITHFUL"
        },
        {
            "name": "Hallucinated Science (manual)",
            "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and glucose. It occurs in chloroplasts.",
            "generated": "Photosynthesis happens in the mitochondria where plants use nitrogen and UV radiation to produce carbon dioxide.",
            "expected": "HALLUCINATED"
        },
        {
            "name": "Faithful History (manual)",
            "context": "The French Revolution began in 1789 with the storming of the Bastille. King Louis XVI was executed in 1793.",
            "generated": "The French Revolution started in 1789 when the Bastille was stormed. Louis XVI was executed in 1793.",
            "expected": "FAITHFUL"
        },
        {
            "name": "Fabricated Facts (manual)",
            "context": "The speed of light in vacuum is 299,792,458 m/s. Nothing with mass can travel at or beyond this speed.",
            "generated": "The speed of light is 186,000 km/s. Neutrinos have been observed exceeding this speed in CERN experiments.",
            "expected": "HALLUCINATED"
        }
    ]
    return jsonify(examples)


@app.route("/api/rag/demo", methods=["GET"])
def rag_demo_examples():
    """RAG-mode demo examples — no manual context needed."""
    return jsonify([
        {
            "name": "RAG — Aspirin (faithful)",
            "query": "How does aspirin work?",
            "generated": "Aspirin inhibits COX enzymes to reduce prostaglandins, which is why it relieves pain. At low doses it also prevents platelets from clumping together.",
            "expected": "FAITHFUL"
        },
        {
            "name": "RAG — Aspirin (hallucinated)",
            "query": "How does aspirin work?",
            "generated": "Aspirin works by blocking serotonin receptors in the brain and stimulates endorphin production, which reduces inflammation.",
            "expected": "HALLUCINATED"
        },
        {
            "name": "RAG — Transformer (faithful)",
            "query": "Who invented the transformer architecture?",
            "generated": "The Transformer architecture was introduced by Vaswani et al. in 2017 in the paper Attention Is All You Need. It uses self-attention instead of recurrent networks.",
            "expected": "FAITHFUL"
        },
        {
            "name": "RAG — Everest (hallucinated)",
            "query": "When was Mount Everest first climbed?",
            "generated": "Mount Everest was first summited in 1952 by a Swiss team. It stands at 8,900 metres on the India-Nepal border.",
            "expected": "HALLUCINATED"
        }
    ])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)