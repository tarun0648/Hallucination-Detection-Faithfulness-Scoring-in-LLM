# rag_evaluator.py — RAG-Augmented Hallucination Evaluator
#
# Extends HallucinationEvaluator with automatic context retrieval.
# Instead of requiring a manual reference context, this class:
#   1. Takes the generated text (or a question + generated text)
#   2. Queries the RAG knowledge base to find the best matching context
#   3. Runs the full NLI + Semantic + LLM scoring pipeline on the retrieved context
#   4. Returns HalluScore + which sources were used as context

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import time
import logging
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_retriever import RAGRetriever, RetrievalResult
from evaluator import HallucinationEvaluator

logger = logging.getLogger(__name__)


class RAGHallucinationEvaluator:
    """
    RAG-augmented hallucination detector.

    How it works:
        user_query + generated_text
               ↓
        RAGRetriever.retrieve(query, top_k=3)
               ↓
        retrieved chunks → joined as context string
               ↓
        NLI Scorer + Semantic Scorer + Gemini Judge
               ↓
        HalluScore + verdict + sources cited

    The key advantage: you don't need to manually provide reference text.
    The system fetches it from the knowledge base automatically.
    """

    def __init__(self,
                 api_key:       str  = None,
                 top_k:         int  = 3,
                 skip_nli:      bool = False,
                 skip_llm_judge: bool = False,
                 use_cot:       bool = False,
                 index_path:    str  = None):
        """
        Args:
            api_key:        Gemini API key
            top_k:          Number of knowledge base chunks to retrieve per query
            skip_nli:       Skip NLI scoring
            skip_llm_judge: Skip Gemini scoring (faster, less accurate)
            use_cot:        Use chain-of-thought in LLM judge
            index_path:     Custom path for the FAISS index
        """
        from config import GEMINI_API_KEY
        self.api_key   = api_key or os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY)
        self.top_k     = top_k
        self.use_cot   = use_cot

        logger.info("Initialising RAG knowledge base...")
        self.retriever = RAGRetriever(index_path=index_path)
        self.retriever.build_index()

        logger.info("Initialising scoring pipeline...")
        self.evaluator = HallucinationEvaluator(
            api_key        = self.api_key,
            skip_nli       = skip_nli,
            skip_llm_judge = skip_llm_judge,
            use_cot        = use_cot
        )

        logger.info("RAG Hallucination Evaluator ready.")

    def evaluate(self, generated_text: str,
                 query:           str  = None,
                 domain_filter:   str  = None,
                 manual_context:  str  = None) -> Dict:
        """
        Full RAG-augmented evaluation.

        Args:
            generated_text:  The LLM output to check for hallucinations
            query:           Optional explicit query (if None, uses generated_text as query)
            domain_filter:   Restrict retrieval to a domain (e.g. "science")
            manual_context:  If provided, skip retrieval and use this as context

        Returns:
            Dict with HalluScore result + retrieval metadata
        """
        # ── Step 1: Retrieve context ──────────────────────────────────────
        if manual_context:
            context  = manual_context
            sources  = []
            rag_used = False
        else:
            search_query = query if query else generated_text
            logger.info(f"Retrieving context for query: {search_query[:80]}...")

            retrieval_results = self.retriever.retrieve(
                query         = search_query,
                top_k         = self.top_k,
                domain_filter = domain_filter
            )
            context  = self.retriever.get_context_string(
                retrieval_results, min_score=0.3
            )
            sources  = self._format_sources(retrieval_results)
            rag_used = True

            used_titles = [s['title'] for s in sources if s['score'] >= 0.3]
            logger.info(f"Retrieved {len(retrieval_results)} candidates; "
                        f"using after threshold filter: {used_titles}")

        if not context.strip():
            return {
                "error":          "No context could be retrieved for this query.",
                "generated_text": generated_text,
                "rag_used":       rag_used
            }

        # ── Step 2: Score against retrieved context ───────────────────────
        score_result = self.evaluator.evaluate_single(
            context        = context,
            generated_text = generated_text,
            sample_id      = "rag_eval"
        )

        # ── Step 3: Enrich result with RAG metadata ───────────────────────
        score_result["rag_metadata"] = {
            "rag_used":        rag_used,
            "query_used":      search_query if rag_used else None,
            "sources_used":    sources,
            "context_preview": context[:300] + "..." if len(context) > 300 else context,
            "top_k":           self.top_k,
            "domain_filter":   domain_filter
        }

        return score_result

    def evaluate_with_question(self, question: str,
                                generated_answer: str,
                                domain_filter: str = None) -> Dict:
        """
        Convenience method: retrieve context using the question,
        then score the generated answer against it.

        This is the most common RAG use case:
            Q: "How does aspirin work?"
            A: <LLM-generated answer>
            → system retrieves medical KB articles about aspirin
            → scores the answer for faithfulness
        """
        return self.evaluate(
            generated_text = generated_answer,
            query          = question,
            domain_filter  = domain_filter
        )

    def batch_evaluate(self, items: List[Dict],
                       delay_sec: float = 15.0) -> List[Dict]:
        """
        Evaluate a batch of (query, generated_text) pairs.

        Each item in `items` should have:
            - "generated_text": str  (required)
            - "query":          str  (optional)
            - "domain":         str  (optional)

        Args:
            items:      List of dicts
            delay_sec:  Seconds to wait between items (rate limit protection)
        """
        results = []
        for i, item in enumerate(items):
            logger.info(f"RAG evaluating item {i+1}/{len(items)}...")
            try:
                result = self.evaluate(
                    generated_text = item["generated_text"],
                    query          = item.get("query"),
                    domain_filter  = item.get("domain")
                )
                result["input"] = item
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on item {i}: {e}")
                results.append({"error": str(e), "input": item})

            if i < len(items) - 1:
                time.sleep(delay_sec)

        return results

    def add_to_knowledge_base(self, title: str, content: str,
                               domain: str = "general",
                               source: str = "custom") -> str:
        """
        Add a new document to the RAG knowledge base and rebuild the index.

        Args:
            title:   Document title
            content: Full document text
            domain:  Domain category (science, history, medicine, technology, general)
            source:  Source label (textbook, wikipedia, custom, etc.)

        Returns:
            doc_id of the added document
        """
        doc_id = self.retriever.add_document(title, content, domain, source)
        logger.info(f"Rebuilding index after adding '{title}'...")
        self.retriever.build_index(force_rebuild=True)
        return doc_id

    def get_knowledge_base_stats(self) -> Dict:
        """Return statistics about the knowledge base."""
        return self.retriever.get_index_stats()

    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the knowledge base without running hallucination scoring.
        Useful for exploring what's in the KB.
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        return [
            {
                "rank":      r.rank,
                "score":     round(r.score, 4),
                "title":     r.chunk.doc_title,
                "domain":    r.chunk.domain,
                "source":    r.chunk.source,
                "preview":   r.chunk.text[:200] + "..."
            }
            for r in results
        ]

    @staticmethod
    def _format_sources(results: List[RetrievalResult]) -> List[Dict]:
        return [
            {
                "rank":    r.rank,
                "score":   round(r.score, 4),
                "title":   r.chunk.doc_title,
                "domain":  r.chunk.domain,
                "source":  r.chunk.source,
                "chunk_id": r.chunk.chunk_id
            }
            for r in results
        ]


# ── CLI demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="RAG Hallucination Evaluator")
    parser.add_argument("--api-key",    type=str, default=None)
    parser.add_argument("--question",   type=str, help="Question asked of the LLM")
    parser.add_argument("--answer",     type=str, help="LLM-generated answer to check")
    parser.add_argument("--domain",     type=str, default=None)
    parser.add_argument("--search",     type=str, help="Search KB only (no scoring)")
    parser.add_argument("--stats",      action="store_true", help="Show KB stats")
    parser.add_argument("--skip-llm",   action="store_true")
    args = parser.parse_args()

    rag_eval = RAGHallucinationEvaluator(
        api_key        = args.api_key,
        skip_llm_judge = args.skip_llm
    )

    if args.stats:
        stats = rag_eval.get_knowledge_base_stats()
        print(json.dumps(stats, indent=2))

    elif args.search:
        results = rag_eval.search_knowledge_base(args.search)
        for r in results:
            print(f"\n[{r['rank']}] {r['title']} (score={r['score']})")
            print(f"    {r['preview'][:120]}...")

    elif args.question and args.answer:
        result = rag_eval.evaluate_with_question(
            question         = args.question,
            generated_answer = args.answer,
            domain_filter    = args.domain
        )
        s = result.get("summary", {})
        meta = result.get("rag_metadata", {})
        print(f"\nHalluScore:  {s.get('hallu_score', 'N/A'):.4f}")
        print(f"Verdict:     {s.get('verdict', 'N/A')}")
        print(f"Confidence:  {s.get('confidence', 'N/A'):.4f}")
        print(f"Sources used:")
        for src in meta.get("sources_used", []):
            print(f"  [{src['rank']}] {src['title']} (sim={src['score']})")
    else:
        parser.print_help()