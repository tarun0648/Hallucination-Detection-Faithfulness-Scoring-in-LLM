# evaluator.py — Main Evaluation Pipeline
#
# Orchestrates the full hallucination detection pipeline:
#   1. Load benchmark
#   2. Run NLI scoring
#   3. Run semantic scoring
#   4. Run LLM judge scoring
#   5. Compute HalluScore
#   6. Evaluate against ground truth labels
#   7. Save results

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from config import (
    GEMINI_API_KEY, GEMINI_MODEL, NLI_MODEL_NAME,
    SENTENCE_MODEL_NAME, RESULTS_DIR,
    WEIGHT_NLI, WEIGHT_SEMANTIC, WEIGHT_LLM_JUDGE, WEIGHT_FACTUALITY,
    FAITHFULNESS_THRESHOLD
)
from nli_scorer     import NLIScorer
from semantic_scorer import SemanticScorer
from llm_judge      import GeminiJudge
from hallu_score    import HalluScoreCalculator, HalluScoreResult
from benchmark      import BenchmarkLoader, BenchmarkSample

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class HallucinationEvaluator:
    """
    Full hallucination detection and faithfulness scoring pipeline.
    
    Usage:
        evaluator = HallucinationEvaluator(api_key="YOUR_GEMINI_KEY")
        results   = evaluator.run_benchmark()
        evaluator.save_results(results)
    """

    def __init__(self,
                 api_key:        str  = None,
                 skip_nli:       bool = False,
                 skip_llm_judge: bool = False,
                 use_cot:        bool = False):
        """
        Initialize the evaluation pipeline.
        
        Args:
            api_key:        Gemini API key (falls back to config.py)
            skip_nli:       Skip NLI scoring (faster, less accurate)
            skip_llm_judge: Skip LLM judge (no API calls needed)
            use_cot:        Use chain-of-thought for LLM judge (slower, more accurate)
        """
        self.api_key        = api_key or GEMINI_API_KEY
        self.skip_nli       = skip_nli
        self.skip_llm_judge = skip_llm_judge
        self.use_cot        = use_cot

        logger.info("Initializing Hallucination Evaluation Pipeline...")

        # Load scorers
        if not skip_nli:
            logger.info("Loading NLI scorer (DeBERTa-v3)...")
            self.nli_scorer = NLIScorer(NLI_MODEL_NAME)

        logger.info("Loading semantic scorer (Sentence Transformers)...")
        self.semantic_scorer = SemanticScorer(SENTENCE_MODEL_NAME)

        if not skip_llm_judge:
            logger.info("Initializing Gemini judge...")
            self.llm_judge = GeminiJudge(self.api_key, GEMINI_MODEL)

        self.hallu_calculator = HalluScoreCalculator(
            weight_nli        = WEIGHT_NLI,
            weight_semantic   = WEIGHT_SEMANTIC,
            weight_llm_judge  = WEIGHT_LLM_JUDGE,
            weight_factuality = WEIGHT_FACTUALITY
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        logger.info("Pipeline ready.")

    def evaluate_single(self, context: str, generated_text: str,
                        sample_id: str = "custom") -> Dict:
        """
        Run full evaluation on a single (context, generated_text) pair.
        
        Returns comprehensive scoring dict.
        """
        result = {"id": sample_id, "context": context[:300],
                  "generated_text": generated_text[:500]}

        # 1. NLI scoring
        if not self.skip_nli:
            logger.info("Running NLI scoring...")
            nli_result = self.nli_scorer.score_document(context, generated_text)
            result["nli"] = nli_result
        else:
            nli_result = {"faithfulness_score": 0.5, "hallucination_rate": 0.0}
            result["nli"] = nli_result

        # 2. Semantic scoring
        logger.info("Running semantic scoring...")
        semantic_result = self.semantic_scorer.score_document(context, generated_text)
        result["semantic"] = semantic_result

        # 3. LLM judge scoring
        if not self.skip_llm_judge:
            logger.info("Running LLM judge scoring...")
            llm_result = self.llm_judge.score_document(
                context, generated_text, use_cot=self.use_cot
            )
            result["llm_judge"] = llm_result
        else:
            llm_result = {"faithfulness_score": 0.5, "verdict": "UNCERTAIN",
                          "confidence": 0.5}
            result["llm_judge"] = llm_result

        # 4. HalluScore
        logger.info("Computing HalluScore...")
        hallu_result = self.hallu_calculator.compute(
            nli_result, semantic_result, llm_result,
            model_name=sample_id
        )
        result["hallu_score_result"] = hallu_result.to_dict()

        # Summary
        result["summary"] = {
            "hallu_score":        hallu_result.hallu_score,
            "verdict":            hallu_result.verdict,
            "confidence":         hallu_result.confidence,
            "nli_score":          hallu_result.nli_score,
            "semantic_score":     hallu_result.semantic_score,
            "llm_judge_score":    hallu_result.llm_judge_score,
            "hallucination_rate": hallu_result.hallucination_rate,
            "is_hallucinated":    hallu_result.verdict == "HALLUCINATED"
        }

        return result

    def run_benchmark(self, benchmark_path: str = None,
                      max_samples: int = None) -> Dict:
        """
        Run full benchmark evaluation.
        
        Args:
            benchmark_path: Path to custom benchmark JSON (None = use built-in)
            max_samples:    Limit samples for testing (None = all)
            
        Returns:
            Complete evaluation results with metrics
        """
        loader  = BenchmarkLoader(benchmark_path)
        samples = list(loader)
        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Running benchmark on {len(samples)} samples...")
        logger.info(f"Benchmark stats: {loader.get_statistics()}")

        all_results = []
        predictions = []  # our predicted labels
        true_labels = []  # ground truth labels

        for sample in tqdm(samples, desc="Evaluating samples"):
            try:
                result = self.evaluate_single(
                    context        = sample.context,
                    generated_text = sample.generated_text,
                    sample_id      = sample.id
                )

                # Add ground truth
                result["ground_truth"] = {
                    "is_faithful":        sample.is_faithful,
                    "hallucination_type": sample.hallucination_type,
                    "domain":             sample.domain,
                    "difficulty":         sample.difficulty,
                    "question":           sample.question,
                    "ground_truth_text":  sample.ground_truth
                }

                all_results.append(result)

                # Collect for metrics
                predicted_faithful = result["summary"]["verdict"] != "HALLUCINATED"
                predictions.append(1 if predicted_faithful else 0)
                true_labels.append(1 if sample.is_faithful else 0)

                logger.info(f"[{sample.id}] HalluScore={result['summary']['hallu_score']:.3f} "
                            f"Verdict={result['summary']['verdict']} "
                            f"GT={'FAITHFUL' if sample.is_faithful else 'HALLUCINATED'}")

                # Respect free-tier rate limit: 5 RPM on gemini-1.5-flash
                # Each sample uses ~2-3 Gemini calls → wait 15s between samples
                if not self.skip_llm_judge:
                    time.sleep(65)

            except Exception as e:
                logger.error(f"Failed on sample {sample.id}: {e}")
                continue

        # Compute aggregate metrics
        metrics = self._compute_metrics(predictions, true_labels, all_results)

        final_output = {
            "benchmark_stats":   loader.get_statistics(),
            "evaluation_metrics": metrics,
            "per_sample_results": all_results,
            "timestamp":         time.strftime("%Y-%m-%d %Human:%M:%S"),
            "config": {
                "nli_model":       NLI_MODEL_NAME,
                "semantic_model":  SENTENCE_MODEL_NAME,
                "llm_judge_model": GEMINI_MODEL,
                "weights": {
                    "nli": WEIGHT_NLI, "semantic": WEIGHT_SEMANTIC,
                    "llm_judge": WEIGHT_LLM_JUDGE, "factuality": WEIGHT_FACTUALITY
                },
                "threshold": FAITHFULNESS_THRESHOLD
            }
        }

        return final_output

    def _compute_metrics(self, predictions: List[int],
                          true_labels: List[int],
                          results: List[Dict]) -> Dict:
        """Compute detection performance metrics."""
        if len(predictions) < 2:
            return {"error": "Not enough samples"}

        # Binary classification metrics
        acc  = accuracy_score(true_labels, predictions)
        prec = precision_score(true_labels, predictions, zero_division=0)
        rec  = recall_score(true_labels, predictions, zero_division=0)
        f1   = f1_score(true_labels, predictions, zero_division=0)
        cm   = confusion_matrix(true_labels, predictions).tolist()

        # Score-level metrics
        hallu_scores = [r["summary"]["hallu_score"] for r in results
                        if "summary" in r]
        true_faithful = [1 if r["ground_truth"]["is_faithful"] else 0
                         for r in results if "ground_truth" in r]

        # ROC-AUC
        try:
            auc = roc_auc_score(true_faithful, hallu_scores)
        except Exception:
            auc = 0.5

        # Per-domain breakdown
        domain_results = {}
        for r in results:
            if "ground_truth" not in r:
                continue
            domain = r["ground_truth"]["domain"]
            if domain not in domain_results:
                domain_results[domain] = {"correct": 0, "total": 0}
            predicted_f = r["summary"]["verdict"] != "HALLUCINATED"
            actual_f    = r["ground_truth"]["is_faithful"]
            domain_results[domain]["total"] += 1
            if predicted_f == actual_f:
                domain_results[domain]["correct"] += 1

        domain_accuracy = {
            d: v["correct"] / v["total"]
            for d, v in domain_results.items() if v["total"] > 0
        }

        # Correlation between HalluScore and ground truth
        from scipy.stats import spearmanr, pearsonr
        try:
            spear, _ = spearmanr(hallu_scores, true_faithful)
            pears, _ = pearsonr(hallu_scores, true_faithful)
        except Exception:
            spear, pears = 0.0, 0.0

        return {
            "accuracy":         round(acc, 4),
            "precision":        round(prec, 4),
            "recall":           round(rec, 4),
            "f1_score":         round(f1, 4),
            "roc_auc":          round(auc, 4),
            "confusion_matrix": cm,
            "spearman_r":       round(float(spear), 4),
            "pearson_r":        round(float(pears), 4),
            "domain_accuracy":  domain_accuracy,
            "total_evaluated":  len(predictions),
            "mean_hallu_score_faithful":   round(float(np.mean(
                [s for s, t in zip(hallu_scores, true_faithful) if t == 1]
            )) if any(t == 1 for t in true_faithful) else 0, 4),
            "mean_hallu_score_hallucinated": round(float(np.mean(
                [s for s, t in zip(hallu_scores, true_faithful) if t == 0]
            )) if any(t == 0 for t in true_faithful) else 0, 4),
        }

    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save evaluation results to JSON and CSV."""
        # Save full JSON
        json_path = os.path.join(RESULTS_DIR, filename)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Full results saved to {json_path}")

        # Save summary CSV
        csv_rows = []
        for r in results.get("per_sample_results", []):
            if "summary" not in r or "ground_truth" not in r:
                continue
            row = {
                "id":                r["id"],
                "domain":            r["ground_truth"]["domain"],
                "difficulty":        r["ground_truth"]["difficulty"],
                "hallu_score":       r["summary"]["hallu_score"],
                "verdict":           r["summary"]["verdict"],
                "nli_score":         r["summary"]["nli_score"],
                "semantic_score":    r["summary"]["semantic_score"],
                "llm_judge_score":   r["summary"]["llm_judge_score"],
                "is_faithful_gt":    r["ground_truth"]["is_faithful"],
                "hallucination_type": r["ground_truth"]["hallucination_type"],
                "correct_prediction": (
                    (r["summary"]["verdict"] != "HALLUCINATED") ==
                    r["ground_truth"]["is_faithful"]
                )
            }
            csv_rows.append(row)

        if csv_rows:
            csv_path = os.path.join(RESULTS_DIR, filename.replace(".json", ".csv"))
            pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
            logger.info(f"Summary CSV saved to {csv_path}")

        return json_path


def run_interactive(api_key: str = None):
    """Interactive mode: evaluate custom (context, text) pairs."""
    print("\n" + "="*60)
    print("  HALLUCINATION DETECTION SYSTEM — Interactive Mode")
    print("="*60)

    evaluator = HallucinationEvaluator(api_key=api_key, use_cot=False)

    while True:
        print("\nEnter context (reference text), or 'quit' to exit:")
        context = input("> ").strip()
        if context.lower() == "quit":
            break

        print("Enter the generated text to evaluate:")
        generated = input("> ").strip()

        result = evaluator.evaluate_single(context, generated)
        summary = result["summary"]

        print("\n" + "-"*40)
        print(f"  HalluScore:    {summary['hallu_score']:.4f}")
        print(f"  Verdict:       {summary['verdict']}")
        print(f"  Confidence:    {summary['confidence']:.4f}")
        print(f"  NLI Score:     {summary['nli_score']:.4f}")
        print(f"  Semantic Score:{summary['semantic_score']:.4f}")
        print(f"  LLM Judge:     {summary['llm_judge_score']:.4f}")
        print("-"*40)

        if summary["is_hallucinated"]:
            print("  ⚠️  HALLUCINATION DETECTED")
        else:
            print("  ✓  Output appears FAITHFUL")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hallucination Detection Pipeline")
    parser.add_argument("--mode",        choices=["benchmark", "interactive"],
                        default="benchmark")
    parser.add_argument("--api-key",     type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-nli",    action="store_true")
    parser.add_argument("--skip-llm",    action="store_true")
    parser.add_argument("--use-cot",     action="store_true")
    args = parser.parse_args()

    if args.mode == "interactive":
        run_interactive(api_key=args.api_key)
    else:
        evaluator = HallucinationEvaluator(
            api_key        = args.api_key,
            skip_nli       = args.skip_nli,
            skip_llm_judge = args.skip_llm,
            use_cot        = args.use_cot
        )
        results = evaluator.run_benchmark(max_samples=args.max_samples)
        path    = evaluator.save_results(results)

        metrics = results["evaluation_metrics"]
        print("\n" + "="*50)
        print("  BENCHMARK RESULTS")
        print("="*50)
        print(f"  Accuracy:   {metrics.get('accuracy', 'N/A')}")
        print(f"  F1 Score:   {metrics.get('f1_score', 'N/A')}")
        print(f"  ROC-AUC:    {metrics.get('roc_auc', 'N/A')}")
        print(f"  Spearman-r: {metrics.get('spearman_r', 'N/A')}")
        print(f"  Results saved to: {path}")