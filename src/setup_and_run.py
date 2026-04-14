#!/usr/bin/env python3
"""
setup_and_run.py — One-script setup and demo runner for HalluDetect
Run this first to verify your environment is set up correctly.
"""

import os, sys, subprocess

def install_deps():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "transformers", "torch", "sentence-transformers",
        "google-generativeai", "flask", "flask-cors",
        "pandas", "numpy", "scikit-learn", "matplotlib",
        "seaborn", "scipy", "nltk", "tqdm", "plotly",
        "python-dotenv", "evaluate"
    ])
    print("Dependencies installed!")

def download_nltk():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    print("NLTK data downloaded!")

def run_quick_demo(api_key: str):
    """Quick demo without running full benchmark."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

    print("\n" + "="*60)
    print("  HALLUDETECT — QUICK DEMO")
    print("="*60)

    from nli_scorer     import NLIScorer
    from semantic_scorer import SemanticScorer
    from llm_judge      import GeminiJudge
    from hallu_score    import HalluScoreCalculator

    nli_scorer   = NLIScorer()
    sem_scorer   = SemanticScorer()
    llm_judge    = GeminiJudge(api_key=api_key)
    calculator   = HalluScoreCalculator()

    pairs = [
        (
            "Photosynthesis occurs in chloroplasts and uses sunlight, water, and CO2 to produce glucose and oxygen.",
            "Plants convert sunlight, water, and CO2 into glucose and oxygen in chloroplasts.",
            "FAITHFUL"
        ),
        (
            "Photosynthesis occurs in chloroplasts and uses sunlight, water, and CO2 to produce glucose and oxygen.",
            "Photosynthesis occurs in mitochondria using nitrogen and UV radiation to produce carbon dioxide.",
            "HALLUCINATED"
        )
    ]

    for ctx, gen, expected in pairs:
        print(f"\n  Expected: {expected}")
        print(f"  Context:  {ctx[:60]}...")
        print(f"  Generated: {gen[:60]}...")

        nli_r  = nli_scorer.score_document(ctx, gen)
        sem_r  = sem_scorer.score_document(ctx, gen)
        llm_r  = llm_judge.score_document(ctx, gen)
        hs     = calculator.compute(nli_r, sem_r, llm_r)

        icon   = "✓" if (hs.verdict == expected) else "✗"
        print(f"  {icon} HalluScore={hs.hallu_score:.4f}  Verdict={hs.verdict}")

    print("\n" + "="*60)
    print("  Quick demo complete! Run the web dashboard with:")
    print("  cd src && python app.py")
    print("="*60 + "\n")

def start_dashboard():
    """Start the Flask web dashboard."""
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    os.chdir(src_dir)
    os.execv(sys.executable, [sys.executable, "app.py"])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--install",    action="store_true", help="Install dependencies")
    parser.add_argument("--demo",       action="store_true", help="Run quick demo")
    parser.add_argument("--dashboard",  action="store_true", help="Start web dashboard")
    parser.add_argument("--api-key",    type=str, default=os.environ.get("GEMINI_API_KEY", ""))
    args = parser.parse_args()

    if args.install:
        install_deps()
        download_nltk()

    if args.demo:
        if not args.api_key:
            print("Error: --api-key required for demo. Set GEMINI_API_KEY env var or pass --api-key.")
            sys.exit(1)
        run_quick_demo(args.api_key)

    if args.dashboard:
        start_dashboard()

    if not any([args.install, args.demo, args.dashboard]):
        print("Usage:")
        print("  python setup_and_run.py --install                          # Install deps")
        print("  python setup_and_run.py --demo --api-key YOUR_KEY          # Quick demo")
        print("  python setup_and_run.py --dashboard                        # Start web UI")
        print("  cd src && python evaluator.py --api-key YOUR_KEY           # Full benchmark")
