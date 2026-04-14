# benchmark.py — Custom Benchmark Dataset for Hallucination Detection
#
# Creates a curated dataset of (context, generated_text, ground_truth) triples
# with known faithful and hallucinated examples across multiple domains.

import json
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single benchmark evaluation sample."""
    id:              str
    domain:          str     # science, history, medicine, general
    context:         str     # reference text / source document
    question:        str     # question asked of the LLM
    generated_text:  str     # LLM's response
    ground_truth:    str     # correct answer
    is_faithful:     bool    # True if generated_text is faithful to context
    hallucination_type: Optional[str]  # type of hallucination if any
    difficulty:      str     # easy / medium / hard
    source:          str     # where this sample came from

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── Hand-curated benchmark samples ─────────────────────────────────────────
BENCHMARK_DATA = [
    # ── SCIENCE domain ──────────────────────────────────────────────────────
    {
        "id": "sci_001",
        "domain": "science",
        "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose. The process occurs in the chloroplasts and involves two stages: the light-dependent reactions and the Calvin cycle.",
        "question": "Explain how photosynthesis works.",
        "generated_text": "Photosynthesis is when plants use sunlight, water, and CO2 to make glucose and oxygen. It happens in the chloroplasts through two stages: light reactions and the Calvin cycle.",
        "ground_truth": "Plants use sunlight, water, and CO2 to produce glucose and oxygen in chloroplasts via light-dependent reactions and Calvin cycle.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "sci_002",
        "domain": "science",
        "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose. The process occurs in the chloroplasts and involves two stages: the light-dependent reactions and the Calvin cycle.",
        "question": "Explain how photosynthesis works.",
        "generated_text": "Photosynthesis occurs in the mitochondria of plant cells. Plants absorb nitrogen from the air and convert it to glucose using ultraviolet radiation. The process releases carbon dioxide as a byproduct.",
        "ground_truth": "Occurs in chloroplasts; uses sunlight, water, CO2; produces oxygen and glucose.",
        "is_faithful": False,
        "hallucination_type": "contradictory",
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "sci_003",
        "domain": "science",
        "context": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3×10^8 m/s). This constant, denoted as 'c', is fundamental to Einstein's theory of special relativity. Nothing with mass can reach or exceed the speed of light.",
        "question": "What is the speed of light?",
        "generated_text": "The speed of light in vacuum is exactly 299,792,458 m/s, denoted as 'c'. According to Einstein's special relativity, objects with mass cannot reach this speed.",
        "ground_truth": "~299,792,458 m/s; constant c; nothing with mass can reach it.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "sci_004",
        "domain": "science",
        "context": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 3×10^8 m/s). This constant, denoted as 'c', is fundamental to Einstein's theory of special relativity. Nothing with mass can reach or exceed the speed of light.",
        "question": "What is the speed of light and can we exceed it?",
        "generated_text": "The speed of light is 186,000 miles per second in vacuum. Recent experiments at CERN in 2022 demonstrated that neutrinos can travel at 1.3 times the speed of light under certain quantum conditions.",
        "ground_truth": "~299,792,458 m/s; cannot be exceeded by matter.",
        "is_faithful": False,
        "hallucination_type": "fabricated",
        "difficulty": "medium",
        "source": "custom"
    },
    # ── HISTORY domain ───────────────────────────────────────────────────────
    {
        "id": "hist_001",
        "domain": "history",
        "context": "The French Revolution began in 1789 with the storming of the Bastille on July 14. It led to the abolition of the French monarchy and the execution of King Louis XVI in January 1793. The Revolution gave rise to Napoleon Bonaparte, who became Emperor in 1804.",
        "question": "When did the French Revolution begin and what were its key events?",
        "generated_text": "The French Revolution started in 1789 when the Bastille was stormed on July 14th. This led to the end of the monarchy, the execution of Louis XVI in 1793, and eventually Napoleon becoming Emperor in 1804.",
        "ground_truth": "1789 Bastille storming; monarchy abolished; Louis XVI executed 1793; Napoleon Emperor 1804.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "hist_002",
        "domain": "history",
        "context": "The French Revolution began in 1789 with the storming of the Bastille on July 14. It led to the abolition of the French monarchy and the execution of King Louis XVI in January 1793. The Revolution gave rise to Napoleon Bonaparte, who became Emperor in 1804.",
        "question": "Describe the French Revolution.",
        "generated_text": "The French Revolution began in 1787 with widespread protests against King Louis XV. The Bastille was stormed in August 1789. Louis XVI was exiled to England rather than executed. Napoleon took power in 1799 as a general, never becoming Emperor.",
        "ground_truth": "Began 1789; Louis XVI (not XV) executed; Napoleon became Emperor 1804.",
        "is_faithful": False,
        "hallucination_type": "contradictory",
        "difficulty": "medium",
        "source": "custom"
    },
    # ── MEDICINE domain ──────────────────────────────────────────────────────
    {
        "id": "med_001",
        "domain": "medicine",
        "context": "Diabetes mellitus is a metabolic disease characterized by high blood sugar. Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin. Type 2 diabetes occurs when the body doesn't use insulin effectively. Both types require different management approaches.",
        "question": "What is the difference between Type 1 and Type 2 diabetes?",
        "generated_text": "Type 1 diabetes is autoimmune: the pancreas makes little or no insulin. Type 2 occurs when the body can't use insulin effectively. Both are types of diabetes mellitus but require different management.",
        "ground_truth": "Type 1: autoimmune, no insulin production. Type 2: insulin resistance. Different management.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "med_002",
        "domain": "medicine",
        "context": "Aspirin (acetylsalicylic acid) works by inhibiting cyclooxygenase (COX) enzymes, which reduces the production of prostaglandins. This leads to its anti-inflammatory, analgesic, and antipyretic effects. At low doses, aspirin also prevents platelet aggregation, reducing clot formation.",
        "question": "How does aspirin work?",
        "generated_text": "Aspirin inhibits COX enzymes to reduce prostaglandin production, giving it anti-inflammatory and pain-relieving properties. At low doses, it prevents platelets from clumping together, which is why it's used to prevent clots.",
        "ground_truth": "Inhibits COX enzymes → reduces prostaglandins → anti-inflammatory, analgesic; low dose prevents platelet aggregation.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "medium",
        "source": "custom"
    },
    {
        "id": "med_003",
        "domain": "medicine",
        "context": "Aspirin (acetylsalicylic acid) works by inhibiting cyclooxygenase (COX) enzymes, which reduces the production of prostaglandins. This leads to its anti-inflammatory, analgesic, and antipyretic effects. At low doses, aspirin also prevents platelet aggregation, reducing clot formation.",
        "question": "Explain the mechanism of aspirin.",
        "generated_text": "Aspirin works by blocking serotonin receptors in the brain, which reduces pain signals. It also stimulates the production of endorphins, which explain its anti-inflammatory effects. High doses of aspirin activate the immune system to prevent blood clots.",
        "ground_truth": "Inhibits COX enzymes; reduces prostaglandins; prevents platelet aggregation.",
        "is_faithful": False,
        "hallucination_type": "fabricated",
        "difficulty": "hard",
        "source": "custom"
    },
    # ── TECHNOLOGY domain ────────────────────────────────────────────────────
    {
        "id": "tech_001",
        "domain": "technology",
        "context": "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al., revolutionized natural language processing. It uses self-attention mechanisms instead of recurrent networks, allowing parallel processing of sequences and capturing long-range dependencies.",
        "question": "What is the transformer architecture?",
        "generated_text": "The transformer architecture was introduced in 'Attention Is All You Need' (Vaswani et al., 2017). It uses self-attention instead of RNNs, enabling parallel processing and better capture of long-range dependencies in sequences.",
        "ground_truth": "2017 paper by Vaswani et al.; uses self-attention; parallel processing; long-range dependencies.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "medium",
        "source": "custom"
    },
    {
        "id": "tech_002",
        "domain": "technology",
        "context": "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need' by Vaswani et al., revolutionized natural language processing. It uses self-attention mechanisms instead of recurrent networks, allowing parallel processing of sequences and capturing long-range dependencies.",
        "question": "Who created the transformer architecture and when?",
        "generated_text": "The transformer was invented by Geoffrey Hinton and his team at Google Brain in 2015. The paper 'Attention Is All You Need' was published in 2019. Transformers use LSTM networks with attention gates to process language sequentially.",
        "ground_truth": "Vaswani et al., 2017; self-attention (not LSTM); parallel not sequential.",
        "is_faithful": False,
        "hallucination_type": "contradictory",
        "difficulty": "medium",
        "source": "custom"
    },
    {
        "id": "tech_003",
        "domain": "technology",
        "context": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Known for its readable syntax and versatility, Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        "question": "Tell me about Python programming language.",
        "generated_text": "Python is a high-level interpreted language by Guido van Rossum, released in 1991. It's known for clean syntax and supports procedural, OOP, and functional paradigms.",
        "ground_truth": "High-level, interpreted; created by Guido van Rossum; released 1991; multi-paradigm.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    # ── GEOGRAPHY domain ─────────────────────────────────────────────────────
    {
        "id": "geo_001",
        "domain": "geography",
        "context": "Mount Everest, located in the Himalayas on the border between Nepal and Tibet, is Earth's highest mountain at 8,848.86 meters (29,031.7 feet) above sea level. It was first summited on May 29, 1953 by Edmund Hillary and Tenzing Norgay.",
        "question": "What is Mount Everest and where is it?",
        "generated_text": "Mount Everest is the world's highest mountain at 8,848.86 meters, on the Nepal-Tibet border in the Himalayas. It was first climbed on May 29, 1953 by Edmund Hillary and Tenzing Norgay.",
        "ground_truth": "8,848.86m; Nepal-Tibet border; first summit May 29, 1953 by Hillary and Norgay.",
        "is_faithful": True,
        "hallucination_type": None,
        "difficulty": "easy",
        "source": "custom"
    },
    {
        "id": "geo_002",
        "domain": "geography",
        "context": "Mount Everest, located in the Himalayas on the border between Nepal and Tibet, is Earth's highest mountain at 8,848.86 meters (29,031.7 feet) above sea level. It was first summited on May 29, 1953 by Edmund Hillary and Tenzing Norgay.",
        "question": "Who first climbed Everest?",
        "generated_text": "Mount Everest was first summited in 1952 by a Swiss expedition led by Raymond Lambert and Tenzing Norgay, who reached the top on October 3rd. The mountain stands at 8,900 meters on the India-Nepal border.",
        "ground_truth": "Hillary and Norgay, May 29, 1953; 8,848.86m; Nepal-Tibet border.",
        "is_faithful": False,
        "hallucination_type": "contradictory",
        "difficulty": "medium",
        "source": "custom"
    },
    # ── UNSUPPORTED claims ───────────────────────────────────────────────────
    {
        "id": "unsup_001",
        "domain": "general",
        "context": "The Amazon River is the largest river in the world by discharge volume of water. It flows through South America, primarily through Brazil. The river basin contains the Amazon rainforest, which is the world's largest tropical rainforest.",
        "question": "Tell me about the Amazon River.",
        "generated_text": "The Amazon River is the largest by water discharge, flowing mainly through Brazil in South America. Its basin contains the world's largest tropical rainforest. The Amazon was named after female warriors by Spanish explorer Francisco de Orellana in 1541 after a legendary battle.",
        "ground_truth": "Largest by discharge; flows through Brazil; Amazon rainforest basin.",
        "is_faithful": False,
        "hallucination_type": "unsupported",
        "difficulty": "hard",
        "source": "custom"
    },
]


class BenchmarkLoader:
    """Loads and manages the hallucination detection benchmark."""

    def __init__(self, benchmark_path: str = None):
        self.samples = []
        if benchmark_path and os.path.exists(benchmark_path):
            self.load_from_file(benchmark_path)
        else:
            self.load_builtin()

    def load_builtin(self):
        """Load the built-in curated benchmark."""
        self.samples = [BenchmarkSample(**s) for s in BENCHMARK_DATA]
        logger.info(f"Loaded {len(self.samples)} built-in benchmark samples.")

    def load_from_file(self, path: str):
        """Load benchmark from JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.samples = [BenchmarkSample(**s) for s in data]
        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def save_to_file(self, path: str):
        """Save benchmark to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([s.to_dict() for s in self.samples], f, indent=2)
        logger.info(f"Saved benchmark to {path}")

    def get_by_domain(self, domain: str) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.domain == domain]

    def get_faithful(self) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.is_faithful]

    def get_hallucinated(self) -> List[BenchmarkSample]:
        return [s for s in self.samples if not s.is_faithful]

    def get_by_difficulty(self, difficulty: str) -> List[BenchmarkSample]:
        return [s for s in self.samples if s.difficulty == difficulty]

    def get_statistics(self) -> Dict:
        total     = len(self.samples)
        faithful  = sum(1 for s in self.samples if s.is_faithful)
        domains   = {}
        hall_types = {}

        for s in self.samples:
            domains[s.domain] = domains.get(s.domain, 0) + 1
            if s.hallucination_type:
                hall_types[s.hallucination_type] = \
                    hall_types.get(s.hallucination_type, 0) + 1

        return {
            "total_samples":        total,
            "faithful_samples":     faithful,
            "hallucinated_samples": total - faithful,
            "balance_ratio":        faithful / total if total > 0 else 0,
            "domains":              domains,
            "hallucination_types":  hall_types,
            "difficulty_breakdown": {
                d: sum(1 for s in self.samples if s.difficulty == d)
                for d in ["easy", "medium", "hard"]
            }
        }

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
