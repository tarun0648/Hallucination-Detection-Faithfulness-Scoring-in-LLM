# nli_scorer.py — Natural Language Inference-based Faithfulness Scorer
# Uses DeBERTa-v3 NLI model to score entailment between context and claims.

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class NLIScorer:
    """
    Scores faithfulness using a cross-encoder NLI model.
    
    For each (context, claim) pair, predicts:
      - ENTAILMENT  → claim is supported by context
      - NEUTRAL     → claim is neither supported nor contradicted
      - CONTRADICTION → claim contradicts context
    
    Returns a faithfulness score in [0, 1] where 1 = fully faithful.
    """

    LABEL_MAP = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base"):
        logger.info(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"NLI model loaded on {self.device}")

    def _predict_single(self, premise: str, hypothesis: str) -> Dict:
        """Run NLI inference on a single premise-hypothesis pair."""
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        # DeBERTa NLI label order: contradiction=0, neutral=1, entailment=2
        return {
            "contradiction_prob": float(probs[0]),
            "neutral_prob":       float(probs[1]),
            "entailment_prob":    float(probs[2]),
            "predicted_label":    self.LABEL_MAP[int(np.argmax(probs))],
            "faithfulness_score": float(probs[2])  # entailment prob = faithfulness
        }

    def score_claim(self, context: str, claim: str) -> Dict:
        """
        Score a single claim against a context.
        
        Args:
            context: The reference/ground-truth text
            claim:   The LLM-generated statement to verify
            
        Returns:
            Dict with NLI probabilities and faithfulness score
        """
        result = self._predict_single(premise=context, hypothesis=claim)
        result["context"] = context[:200] + "..." if len(context) > 200 else context
        result["claim"]   = claim
        return result

    def score_document(self, context: str, generated_text: str) -> Dict:
        """
        Score a full generated document against a context by splitting into sentences.
        
        Returns sentence-level scores + document-level aggregate.
        """
        import nltk
        try:
            sentences = nltk.sent_tokenize(generated_text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(generated_text)

        if not sentences:
            return {"faithfulness_score": 0.0, "sentence_scores": []}

        sentence_scores = []
        for sent in sentences:
            if len(sent.strip()) < 10:  # skip trivially short sentences
                continue
            score = self._predict_single(premise=context, hypothesis=sent)
            score["sentence"] = sent
            sentence_scores.append(score)

        if not sentence_scores:
            return {"faithfulness_score": 1.0, "sentence_scores": []}

        # Document-level score = mean entailment probability across sentences
        faithfulness_scores = [s["faithfulness_score"] for s in sentence_scores]
        hallucinated_count  = sum(1 for s in sentence_scores
                                  if s["predicted_label"] == "CONTRADICTION")
        uncertain_count     = sum(1 for s in sentence_scores
                                  if s["predicted_label"] == "NEUTRAL")

        return {
            "faithfulness_score":     float(np.mean(faithfulness_scores)),
            "min_faithfulness":       float(np.min(faithfulness_scores)),
            "max_faithfulness":       float(np.max(faithfulness_scores)),
            "std_faithfulness":       float(np.std(faithfulness_scores)),
            "hallucinated_sentences": hallucinated_count,
            "uncertain_sentences":    uncertain_count,
            "faithful_sentences":     len(sentence_scores) - hallucinated_count - uncertain_count,
            "total_sentences":        len(sentence_scores),
            "hallucination_rate":     hallucinated_count / len(sentence_scores),
            "sentence_scores":        sentence_scores
        }

    def batch_score(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Dict]:
        """
        Batch score multiple (context, claim) pairs efficiently.
        
        Args:
            pairs: List of (context, claim) tuples
            batch_size: Number of pairs to process at once
            
        Returns:
            List of score dicts
        """
        results = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            premises    = [p[0] for p in batch]
            hypotheses  = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises, hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, probs in enumerate(probs_batch):
                results.append({
                    "context":             premises[j],
                    "claim":               hypotheses[j],
                    "contradiction_prob":  float(probs[0]),
                    "neutral_prob":        float(probs[1]),
                    "entailment_prob":     float(probs[2]),
                    "predicted_label":     self.LABEL_MAP[int(np.argmax(probs))],
                    "faithfulness_score":  float(probs[2])
                })

        return results