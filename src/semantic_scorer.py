import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os

# semantic_scorer.py — Semantic Similarity-based Faithfulness Scorer
# Uses sentence-transformers to compute embedding similarity between context and claim.

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SemanticScorer:
    """
    Computes semantic similarity between context and generated text
    using sentence-level embeddings.
    
    Captures faithfulness from a different angle than NLI:
    - High similarity → likely faithful
    - Low similarity  → possible topic drift or hallucination
    
    Also computes BERTScore-style precision/recall/F1.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Sentence transformer loaded.")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def score_claim(self, context: str, claim: str) -> Dict:
        """Score a single claim against context using semantic similarity."""
        embeddings = self.model.encode([context, claim], convert_to_numpy=True)
        sim = self.cosine_similarity(embeddings[0], embeddings[1])

        # Normalize to [0, 1] — cosine sim is in [-1, 1]
        normalized = (sim + 1) / 2

        return {
            "cosine_similarity":  float(sim),
            "faithfulness_score": float(normalized),
            "context":            context[:200],
            "claim":              claim
        }

    def bert_score_style(self, context_sentences: List[str],
                         generated_sentences: List[str]) -> Dict:
        """
        BERTScore-style precision, recall, F1 between two lists of sentences.
        
        Precision: for each generated sentence, find max sim with any context sentence
        Recall:    for each context sentence, find max sim with any generated sentence
        F1:        harmonic mean
        """
        if not context_sentences or not generated_sentences:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ctx_embs = self.model.encode(context_sentences, convert_to_numpy=True)
        gen_embs = self.model.encode(generated_sentences, convert_to_numpy=True)

        # Compute all pairwise cosine similarities
        sim_matrix = np.zeros((len(gen_embs), len(ctx_embs)))
        for i, ge in enumerate(gen_embs):
            for j, ce in enumerate(ctx_embs):
                sim_matrix[i, j] = self.cosine_similarity(ge, ce)

        # Precision: generated sentences are grounded in context
        precision = float(np.mean(np.max(sim_matrix, axis=1)))

        # Recall: context is covered by generated text
        recall = float(np.mean(np.max(sim_matrix, axis=0)))

        # F1
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "precision":          precision,
            "recall":             recall,
            "f1":                 f1,
            "faithfulness_score": f1  # F1 as overall faithfulness
        }

    def score_document(self, context: str, generated_text: str) -> Dict:
        """Full document scoring with sentence-level breakdown."""
        import nltk
        try:
            ctx_sents = nltk.sent_tokenize(context)
            gen_sents = nltk.sent_tokenize(generated_text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            ctx_sents = nltk.sent_tokenize(context)
            gen_sents = nltk.sent_tokenize(generated_text)

        # Overall doc similarity
        doc_score = self.score_claim(context, generated_text)

        # BERTScore-style
        bert_scores = self.bert_score_style(ctx_sents, gen_sents)

        # Per-sentence scores for generated text
        sentence_scores = []
        for sent in gen_sents:
            if len(sent.strip()) < 10:
                continue
            s = self.score_claim(context, sent)
            s["sentence"] = sent
            sentence_scores.append(s)

        all_sims = [s["faithfulness_score"] for s in sentence_scores]

        return {
            "document_cosine_similarity":  doc_score["cosine_similarity"],
            "faithfulness_score":          bert_scores["f1"],
            "bertscore_precision":         bert_scores["precision"],
            "bertscore_recall":            bert_scores["recall"],
            "bertscore_f1":                bert_scores["f1"],
            "mean_sentence_similarity":    float(np.mean(all_sims)) if all_sims else 0.0,
            "min_sentence_similarity":     float(np.min(all_sims)) if all_sims else 0.0,
            "sentence_scores":             sentence_scores
        }

    def batch_score(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Batch score multiple (context, claim) pairs."""
        contexts = [p[0] for p in pairs]
        claims   = [p[1] for p in pairs]

        all_texts = contexts + claims
        all_embs  = self.model.encode(all_texts, convert_to_numpy=True, batch_size=32)

        ctx_embs   = all_embs[:len(contexts)]
        claim_embs = all_embs[len(contexts):]

        results = []
        for i in range(len(pairs)):
            sim = self.cosine_similarity(ctx_embs[i], claim_embs[i])
            results.append({
                "context":            contexts[i],
                "claim":              claims[i],
                "cosine_similarity":  float(sim),
                "faithfulness_score": float((sim + 1) / 2)
            })

        return results