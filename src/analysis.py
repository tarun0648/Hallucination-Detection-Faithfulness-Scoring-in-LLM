# analysis.py — Results Analysis and Visualization
#
# Generates publication-quality figures comparing HalluScore against baselines.

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "hallu_score":  "#2563EB",   # blue
    "nli":          "#16A34A",   # green
    "semantic":     "#D97706",   # amber
    "llm_judge":    "#9333EA",   # purple
    "faithful":     "#22C55E",
    "hallucinated": "#EF4444",
    "uncertain":    "#F59E0B"
}


class ResultsAnalyzer:
    """Analyzes and visualizes hallucination detection results."""

    def __init__(self, results_path: str):
        with open(results_path) as f:
            self.results = json.load(f)
        self.samples = self.results.get("per_sample_results", [])
        self.metrics = self.results.get("evaluation_metrics", {})
        self.df      = self._build_dataframe()
        print(f"Loaded {len(self.samples)} samples for analysis.")

    def _build_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.samples:
            if "summary" not in r or "ground_truth" not in r:
                continue
            s  = r["summary"]
            gt = r["ground_truth"]
            rows.append({
                "id":                 r["id"],
                "domain":             gt["domain"],
                "difficulty":         gt["difficulty"],
                "is_faithful_gt":     gt["is_faithful"],
                "hallucination_type": gt.get("hallucination_type"),
                "hallu_score":        s["hallu_score"],
                "nli_score":          s["nli_score"],
                "semantic_score":     s["semantic_score"],
                "llm_judge_score":    s["llm_judge_score"],
                "verdict":            s["verdict"],
                "confidence":         s["confidence"],
                "hallucination_rate": s["hallucination_rate"],
                "correct":            (s["verdict"] != "HALLUCINATED") == gt["is_faithful"]
            })
        return pd.DataFrame(rows)

    def plot_score_distributions(self, save_path: str = "results/score_distributions.png"):
        """Plot score distributions for faithful vs hallucinated samples."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Score Distributions: Faithful vs Hallucinated Samples",
                     fontsize=14, fontweight="bold", y=1.01)

        score_cols = {
            "hallu_score":   "HalluScore (Ours)",
            "nli_score":     "NLI Score (DeBERTa)",
            "semantic_score": "Semantic Score",
            "llm_judge_score": "LLM Judge Score"
        }

        faithful_df    = self.df[self.df["is_faithful_gt"] == True]
        hallucinated_df = self.df[self.df["is_faithful_gt"] == False]

        for ax, (col, title) in zip(axes.flat, score_cols.items()):
            if col not in self.df.columns:
                continue

            ax.hist(faithful_df[col].dropna(), bins=15, alpha=0.7,
                    color=COLORS["faithful"], label="Faithful", edgecolor="white")
            ax.hist(hallucinated_df[col].dropna(), bins=15, alpha=0.7,
                    color=COLORS["hallucinated"], label="Hallucinated", edgecolor="white")

            ax.axvline(faithful_df[col].mean(), color=COLORS["faithful"],
                       linestyle="--", linewidth=2, alpha=0.9)
            ax.axvline(hallucinated_df[col].mean(), color=COLORS["hallucinated"],
                       linestyle="--", linewidth=2, alpha=0.9)

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.legend()
            ax.set_xlim(0, 1)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Score distribution plot saved: {save_path}")

    def plot_radar_metrics(self, save_path: str = "results/metrics_radar.png"):
        """Radar chart comparing all metrics."""
        metrics_data = {
            "HalluScore":    [self.metrics.get("accuracy", 0.5),
                              self.metrics.get("precision", 0.5),
                              self.metrics.get("recall", 0.5),
                              self.metrics.get("f1_score", 0.5),
                              self.metrics.get("roc_auc", 0.5)],
        }

        categories = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.set_facecolor("#F8FAFC")

        colors_list = list(COLORS.values())
        for i, (name, values) in enumerate(metrics_data.items()):
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2,
                    color=colors_list[i], label=name)
            ax.fill(angles, values, alpha=0.15, color=colors_list[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        ax.set_title("Detection Performance Metrics", fontsize=14,
                     fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Radar chart saved: {save_path}")

    def plot_domain_accuracy(self, save_path: str = "results/domain_accuracy.png"):
        """Bar chart of accuracy per domain."""
        domain_acc = self.metrics.get("domain_accuracy", {})
        if not domain_acc:
            return

        domains    = list(domain_acc.keys())
        accuracies = [domain_acc[d] for d in domains]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(domains, accuracies, color=COLORS["hallu_score"],
                      alpha=0.85, edgecolor="white", linewidth=1.5)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{acc:.2f}",
                    ha="center", va="bottom", fontweight="bold")

        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline")
        ax.set_xlabel("Domain", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Hallucination Detection Accuracy by Domain",
                     fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Domain accuracy chart saved: {save_path}")

    def plot_score_correlation(self, save_path: str = "results/score_correlation.png"):
        """Heatmap of correlation between scoring methods."""
        score_cols = ["hallu_score", "nli_score", "semantic_score", "llm_judge_score"]
        available  = [c for c in score_cols if c in self.df.columns]

        if len(available) < 2:
            return

        corr_matrix = self.df[available].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="Blues",
                    vmin=0, vmax=1, ax=ax,
                    xticklabels=["HalluScore", "NLI", "Semantic", "LLM Judge"],
                    yticklabels=["HalluScore", "NLI", "Semantic", "LLM Judge"],
                    linewidths=0.5, linecolor="white")

        ax.set_title("Inter-Scorer Correlation Matrix",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Correlation heatmap saved: {save_path}")

    def plot_confusion_matrix(self, save_path: str = "results/confusion_matrix.png"):
        """Plot confusion matrix."""
        cm = self.metrics.get("confusion_matrix")
        if not cm:
            return

        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Predicted Hall.", "Predicted Faith."],
                    yticklabels=["Actual Hall.", "Actual Faith."],
                    linewidths=1, linecolor="white")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved: {save_path}")

    def plot_hallucination_types(self, save_path: str = "results/hallucination_types.png"):
        """Breakdown of hallucination types and detection rates."""
        hall_df = self.df[self.df["is_faithful_gt"] == False].copy()
        if hall_df.empty:
            return

        type_counts = hall_df["hallucination_type"].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart of types
        ax1.pie(type_counts.values, labels=type_counts.index,
                autopct="%1.1f%%", colors=sns.color_palette("Set2"),
                startangle=90, pctdistance=0.8)
        ax1.set_title("Hallucination Types in Benchmark", fontweight="bold")

        # Detection rate per type
        type_detection = {}
        for h_type in hall_df["hallucination_type"].unique():
            type_data = hall_df[hall_df["hallucination_type"] == h_type]
            detected  = (type_data["verdict"] == "HALLUCINATED").mean()
            type_detection[h_type] = detected

        types  = list(type_detection.keys())
        detect = [type_detection[t] for t in types]

        bars = ax2.bar(types, detect,
                       color=[COLORS["hallucinated"]] * len(types),
                       alpha=0.8, edgecolor="white")
        for bar, d in zip(bars, detect):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{d:.0%}",
                     ha="center", va="bottom", fontweight="bold")

        ax2.set_xlabel("Hallucination Type")
        ax2.set_ylabel("Detection Rate")
        ax2.set_title("Detection Rate by Hallucination Type", fontweight="bold")
        ax2.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Hallucination types chart saved: {save_path}")

    def generate_report(self, output_dir: str = "results/") -> str:
        """Generate a comprehensive text report."""
        metrics = self.metrics
        lines   = [
            "=" * 65,
            "  HALLUCINATION DETECTION SYSTEM — EVALUATION REPORT",
            "=" * 65,
            "",
            "  BENCHMARK STATISTICS",
            f"  Total Samples:          {metrics.get('total_evaluated', 'N/A')}",
            f"  Faithful Samples:       {self.results['benchmark_stats'].get('faithful_samples', 'N/A')}",
            f"  Hallucinated Samples:   {self.results['benchmark_stats'].get('hallucinated_samples', 'N/A')}",
            "",
            "  DETECTION PERFORMANCE",
            f"  Accuracy:               {metrics.get('accuracy', 'N/A')}",
            f"  Precision:              {metrics.get('precision', 'N/A')}",
            f"  Recall:                 {metrics.get('recall', 'N/A')}",
            f"  F1 Score:               {metrics.get('f1_score', 'N/A')}",
            f"  ROC-AUC:                {metrics.get('roc_auc', 'N/A')}",
            "",
            "  SCORE CORRELATION WITH GROUND TRUTH",
            f"  Spearman-r:             {metrics.get('spearman_r', 'N/A')}",
            f"  Pearson-r:              {metrics.get('pearson_r', 'N/A')}",
            "",
            "  SCORE SEPARATION",
            f"  Mean HalluScore (Faithful):     {metrics.get('mean_hallu_score_faithful', 'N/A')}",
            f"  Mean HalluScore (Hallucinated): {metrics.get('mean_hallu_score_hallucinated', 'N/A')}",
            "",
            "  DOMAIN ACCURACY",
        ]

        for domain, acc in metrics.get("domain_accuracy", {}).items():
            lines.append(f"  {domain:20s}: {acc:.4f}")

        lines += [
            "",
            "  COMPONENT WEIGHTS (HalluScore)",
            f"  NLI Weight:             {self.results['config']['weights']['nli']}",
            f"  Semantic Weight:        {self.results['config']['weights']['semantic']}",
            f"  LLM Judge Weight:       {self.results['config']['weights']['llm_judge']}",
            f"  Factuality Weight:      {self.results['config']['weights']['factuality']}",
            "=" * 65,
        ]

        report = "\n".join(lines)
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report)

        print(report)
        print(f"\nReport saved to {report_path}")
        return report_path

    def generate_all_plots(self):
        """Generate all analysis plots."""
        self.plot_score_distributions()
        self.plot_domain_accuracy()
        self.plot_score_correlation()
        self.plot_confusion_matrix()
        self.plot_hallucination_types()
        self.plot_radar_metrics()
        self.generate_report()
        print("\nAll analysis complete. Check the results/ directory.")


if __name__ == "__main__":
    import sys

    # Resolve results path — check multiple locations so it works
    # whether you run from src/, project root, or pass an explicit path.
    _src_dir     = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(_src_dir)

    if len(sys.argv) > 1:
        candidate = sys.argv[1]
    else:
        candidate = "results/benchmark_results.json"

    # Build a list of places to look
    search_paths = [
        candidate,
        os.path.join(_project_dir, candidate),
        os.path.join(_project_dir, "results", "benchmark_results.json"),
        os.path.join(_src_dir,     "results", "benchmark_results.json"),
        os.path.join(os.getcwd(),  candidate),
    ]

    results_path = None
    for p in search_paths:
        if os.path.exists(p):
            results_path = p
            break

    if results_path is None:
        print("Could not find benchmark_results.json. Searched:")
        for p in search_paths:
            print(f"  {p}")
        print("\nRun: python3 evaluator.py --api-key YOUR_KEY  first")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    analyzer = ResultsAnalyzer(results_path)
    analyzer.generate_all_plots()