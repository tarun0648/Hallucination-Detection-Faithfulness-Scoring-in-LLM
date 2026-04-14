import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os

# rag_retriever.py — RAG (Retrieval-Augmented Generation) Knowledge Base
#
# How it fits into HalluDetect:
#   Before: user must manually supply a reference context for every claim.
#   After:  user only supplies the generated text (or a question).
#           RAG retrieves the most relevant context chunks from the knowledge
#           base automatically, then the existing NLI / Semantic / LLM scorers
#           run against that retrieved context.
#
# Architecture:
#   Documents → chunked → embedded (MiniLM) → stored in FAISS index
#   Query     → embedded → top-k nearest chunks retrieved → joined as context

import os
import json
import pickle
import logging
import hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed — falling back to numpy cosine search.")

from sentence_transformers import SentenceTransformer


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Document:
    """A source document added to the knowledge base."""
    doc_id:   str
    title:    str
    content:  str
    source:   str        # e.g. "wikipedia", "textbook", "custom"
    domain:   str        # e.g. "science", "history", "medicine"
    metadata: Dict = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class Chunk:
    """A text chunk derived from a Document."""
    chunk_id:   str
    doc_id:     str
    doc_title:  str
    text:       str
    domain:     str
    source:     str
    chunk_index: int     # position within the document


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its relevance score."""
    chunk:       Chunk
    score:       float   # cosine similarity [0, 1]
    rank:        int


# ─── Built-in knowledge base ─────────────────────────────────────────────────
# A curated set of factual documents covering the benchmark domains.
# You can add more via RAGRetriever.add_document() at runtime.

BUILTIN_DOCUMENTS = [
    {
        "doc_id": "kb_sci_001",
        "title": "Photosynthesis",
        "domain": "science",
        "source": "textbook",
        "content": """Photosynthesis is the process by which plants, algae, and some bacteria use sunlight,
water, and carbon dioxide to produce oxygen and energy in the form of glucose. The overall
equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. The process occurs in the
chloroplasts of plant cells, specifically in structures called thylakoids and the stroma.
Photosynthesis has two main stages: the light-dependent reactions (which occur in the
thylakoid membranes and convert light energy into chemical energy in the form of ATP and
NADPH) and the Calvin cycle (also called the light-independent reactions or the dark
reactions, which occur in the stroma and use ATP and NADPH to fix carbon dioxide into
glucose). Chlorophyll is the primary pigment that absorbs light energy, mainly in the red
and blue wavelengths. The oxygen released during photosynthesis comes from the splitting
of water molecules (photolysis), not from carbon dioxide."""
    },
    {
        "doc_id": "kb_sci_002",
        "title": "Speed of light and special relativity",
        "domain": "science",
        "source": "textbook",
        "content": """The speed of light in a vacuum is exactly 299,792,458 metres per second
(approximately 3×10^8 m/s), denoted by the symbol c. This value is a defined constant in
the International System of Units (SI). According to Einstein's theory of special
relativity (1905), c is the ultimate speed limit of the universe. No object with mass can
reach or exceed the speed of light, because the energy required to accelerate a massive
object approaches infinity as its speed approaches c. Light travels slightly slower in
media other than vacuum; for example, it travels at about 2/3 of c in glass. The concept
that nothing travels faster than light has been confirmed repeatedly in experiments.
Neutrinos were briefly thought to exceed c in a 2011 OPERA experiment, but the result was
later found to be due to a faulty optical fibre cable and a clock oscillator error. All
subsequent experiments have confirmed that neutrinos travel at or very slightly below c."""
    },
    {
        "doc_id": "kb_sci_003",
        "title": "Transformer architecture in deep learning",
        "domain": "technology",
        "source": "research",
        "content": """The Transformer is a deep learning architecture introduced by Vaswani et al. in the
2017 paper 'Attention Is All You Need', published at NeurIPS. It was developed by
researchers at Google Brain and Google Research. The key innovation is the self-attention
mechanism, which allows the model to weigh the importance of different tokens in a sequence
when computing a representation for each token. Unlike recurrent neural networks (RNNs) and
LSTMs, Transformers process all tokens in parallel, making training much faster on modern
hardware. The architecture consists of an encoder stack and a decoder stack, each made up
of layers containing multi-head self-attention and position-wise feed-forward networks.
Layer normalisation and residual connections are used throughout. Transformers have become
the foundation for large language models including BERT, GPT, T5, and many others. The
original paper used 8 attention heads, a model dimension of 512, and was trained on the
WMT English-to-German translation task."""
    },
    {
        "doc_id": "kb_sci_004",
        "title": "Python programming language",
        "domain": "technology",
        "source": "encyclopedia",
        "content": """Python is a high-level, interpreted, general-purpose programming language created by
Guido van Rossum. Development began in the late 1980s and the first version (Python 0.9.0)
was released in February 1991. Python 2.0 was released in 2000 and Python 3.0 in 2008.
Python is designed for code readability and uses significant whitespace (indentation) to
delimit code blocks rather than curly braces or keywords. It supports multiple programming
paradigms including procedural, object-oriented, and functional programming. Python has a
comprehensive standard library and a large ecosystem of third-party packages available
through the Python Package Index (PyPI). It is widely used in web development (Django,
Flask), data science (NumPy, pandas, scikit-learn), machine learning (TensorFlow, PyTorch),
automation, and scientific computing. Python is consistently ranked among the most popular
programming languages in surveys such as Stack Overflow's Developer Survey and the TIOBE
Index."""
    },
    {
        "doc_id": "kb_hist_001",
        "title": "French Revolution",
        "domain": "history",
        "source": "encyclopedia",
        "content": """The French Revolution was a period of radical political and societal transformation in
France that began in 1789 and lasted until 1799. The Revolution overthrew the monarchy,
established a republic, and culminated in Napoleon Bonaparte seizing power. Key events
include: the convening of the Estates-General in May 1789; the Tennis Court Oath on
20 June 1789; the Storming of the Bastille on 14 July 1789 (now France's national day);
the Declaration of the Rights of Man and of the Citizen in August 1789; the abolition of
feudalism; and the execution of King Louis XVI by guillotine on 21 January 1793. The Reign
of Terror (September 1793 – July 1794), led by Robespierre and the Committee of Public
Safety, resulted in approximately 17,000 official death sentences. The Revolution ended
with Napoleon's coup d'état on 9 November 1799 (18 Brumaire). Napoleon was crowned Emperor
of the French on 2 December 1804. The Revolution had profound effects across Europe and
influenced democratic movements worldwide."""
    },
    {
        "doc_id": "kb_med_001",
        "title": "Diabetes mellitus",
        "domain": "medicine",
        "source": "medical_reference",
        "content": """Diabetes mellitus is a group of metabolic diseases characterised by chronic
hyperglycaemia (high blood glucose) resulting from defects in insulin secretion, insulin
action, or both. Type 1 diabetes mellitus (T1DM) is an autoimmune condition in which the
immune system destroys the insulin-producing beta cells of the pancreatic islets of
Langerhans, resulting in little or no insulin production. It typically manifests in
childhood or adolescence and requires lifelong insulin therapy. Type 2 diabetes mellitus
(T2DM) is characterised by insulin resistance (cells fail to respond normally to insulin)
often combined with relatively reduced insulin secretion. T2DM is strongly associated with
obesity, physical inactivity, and family history. It accounts for approximately 90–95% of
all diabetes cases. Treatment involves lifestyle modification, oral antidiabetic drugs
(e.g. metformin), and in many cases insulin. Gestational diabetes occurs during pregnancy.
Chronic complications include cardiovascular disease, nephropathy, retinopathy, and
neuropathy."""
    },
    {
        "doc_id": "kb_med_002",
        "title": "Aspirin mechanism of action",
        "domain": "medicine",
        "source": "medical_reference",
        "content": """Aspirin (acetylsalicylic acid) is a non-steroidal anti-inflammatory drug (NSAID). Its
primary mechanism of action is the irreversible inhibition of cyclooxygenase (COX) enzymes,
specifically COX-1 and COX-2, by acetylating a serine residue in the enzyme's active site.
COX enzymes catalyse the conversion of arachidonic acid to prostaglandins and thromboxanes.
By inhibiting COX, aspirin reduces the synthesis of prostaglandins, which mediate
inflammation, pain, and fever — giving aspirin its anti-inflammatory, analgesic, and
antipyretic properties. At low doses (75–325 mg/day), aspirin preferentially inhibits
COX-1 in platelets. Since platelets lack nuclei and cannot synthesise new COX, the
inhibition lasts the platelet's lifetime (~7–10 days). This prevents the formation of
thromboxane A2, a potent platelet aggregator and vasoconstrictor, thereby reducing
platelet clumping and the risk of arterial thrombosis. This antiplatelet effect is why
low-dose aspirin is used to prevent heart attacks and strokes."""
    },
    {
        "doc_id": "kb_geo_001",
        "title": "Mount Everest",
        "domain": "geography",
        "source": "encyclopedia",
        "content": """Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur
Himal sub-range of the Himalayas on the international border between Nepal and Tibet
Autonomous Region of China. Its summit elevation is 8,848.86 metres (29,031.7 ft) above
sea level, as measured by a 2020 Chinese survey confirmed by Nepal in 2020. Everest was
first summited on 29 May 1953 by New Zealand mountaineer Edmund Hillary and Tibetan-Nepali
Sherpa Tenzing Norgay, as part of a British expedition led by John Hunt. The mountain is
known as Sagarmatha in Nepali and Chomolungma in Tibetan. It was named after George
Everest, the British surveyor-general of India, in 1865. The mountain presents extreme
challenges including the 'death zone' above 8,000 m where oxygen levels are insufficient
to sustain human life for extended periods. Common routes include the Southeast Ridge from
Nepal and the Northeast Ridge from Tibet."""
    },
    {
        "doc_id": "kb_gen_001",
        "title": "Amazon River",
        "domain": "geography",
        "source": "encyclopedia",
        "content": """The Amazon River is the largest river in the world by discharge volume of water,
carrying approximately 20% of all fresh water that flows into the world's oceans. It flows
roughly 6,400 km (3,976 miles) through South America, primarily through Brazil, with
portions in Peru and Colombia. The river basin, known as Amazonia, covers about 7 million
square kilometres and contains the Amazon rainforest, the world's largest tropical
rainforest, which is home to an estimated 10% of all species on Earth. The river was named
after the legendary Amazons, female warriors from Greek mythology. Spanish explorer
Francisco de Orellana led the first European navigation of the full length of the Amazon
in 1541–1542. During his journey, his chronicler Gaspar de Carvajal reported encounters
with female warriors, inspiring Orellana to name the river after the Amazons. The Amazon
has more than 1,100 tributaries, 17 of which are over 1,500 km long."""
    },
    {
        "doc_id": "kb_ai_001",
        "title": "Large language models and hallucination",
        "domain": "technology",
        "source": "research",
        "content": """Large language models (LLMs) are neural networks trained on vast text corpora to
predict and generate human-like text. Hallucination in LLMs refers to the generation of
content that is factually incorrect, unsupported by the input context, or entirely
fabricated, despite appearing fluent and confident. Hallucinations occur because LLMs learn
statistical patterns rather than explicit world knowledge, and they optimise for fluency
rather than factual accuracy. Types of hallucination include: intrinsic hallucination
(contradicting the source context), extrinsic hallucination (introducing information not
present in the source), and factual hallucination (stating falsehoods about the world).
Detection methods include NLI-based entailment checking, self-consistency sampling
(SelfCheckGPT), retrieval-augmented generation (RAG), LLM-as-judge evaluation (G-Eval),
and atomic fact verification (FActScore). Mitigation strategies include retrieval-augmented
generation, RLHF, chain-of-thought prompting, and constrained decoding. Hallucination
remains one of the most active research areas in NLP."""
    },
    {
        "doc_id": "kb_ai_002",
        "title": "Retrieval-Augmented Generation (RAG)",
        "domain": "technology",
        "source": "research",
        "content": """Retrieval-Augmented Generation (RAG) is a framework introduced by Lewis et al. (2020)
at Facebook AI Research that combines parametric knowledge (stored in model weights) with
non-parametric knowledge (retrieved from an external document store). In RAG, a query is
first encoded into a dense vector using a bi-encoder (typically a pre-trained transformer).
This vector is used to retrieve the most relevant documents from a vector database (e.g.
FAISS, Pinecone, ChromaDB) using approximate nearest-neighbour search. The retrieved
documents are then concatenated with the original query and passed to a sequence-to-sequence
language model to generate the final answer. RAG reduces hallucinations by grounding
generation in retrieved facts, enables knowledge updates without retraining the model, and
provides source attribution. The retriever component is typically a dense retriever like
DPR (Dense Passage Retrieval) or a sentence transformer. RAG is widely used for
question-answering, document summarisation, and chatbots requiring up-to-date information."""
    },
    {
        "doc_id": "kb_ai_003",
        "title": "NLI and textual entailment",
        "domain": "technology",
        "source": "research",
        "content": """Natural Language Inference (NLI), also called textual entailment, is the task of
determining the logical relationship between two text snippets: a premise and a hypothesis.
The relationship is classified as: ENTAILMENT (the premise guarantees the truth of the
hypothesis), CONTRADICTION (the premise guarantees the falsity of the hypothesis), or
NEUTRAL (the premise neither entails nor contradicts the hypothesis). NLI models are used
as a proxy for faithfulness detection in summarisation and RAG systems. DeBERTa (Decoding-
enhanced BERT with Disentangled Attention), introduced by He et al. (2021) at Microsoft
Research, achieves state-of-the-art performance on NLI benchmarks including MNLI, SNLI,
and the adversarial NLI dataset. The cross-encoder variant (cross-encoder/nli-deberta-v3-
base) takes the full premise-hypothesis pair as a single sequence and is particularly
suited to document-level faithfulness scoring because it attends to both texts jointly."""
    },
]


# ─── RAG Retriever ────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieval-Augmented Generation knowledge base for HalluDetect.

    Usage:
        retriever = RAGRetriever()
        retriever.build_index()          # embed all built-in docs

        results = retriever.retrieve("How does photosynthesis work?", top_k=3)
        context = retriever.get_context_string(results)
        # → pass context to NLI / Semantic / LLM scorers
    """

    def __init__(self,
                 embed_model:  str = "all-MiniLM-L6-v2",
                 chunk_size:   int = 80,
                 chunk_overlap: int = 20,
                 index_path:   str = None):
        """
        Args:
            embed_model:    Sentence-Transformer model for encoding
            chunk_size:     words per chunk
            chunk_overlap:  overlap between consecutive chunks (words)
            index_path:     path to persist/load FAISS index
        """
        self.embed_model_name = embed_model
        self.chunk_size       = chunk_size
        self.chunk_overlap    = chunk_overlap
        self.index_path       = index_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "rag_index"
        )

        self.encoder: Optional[SentenceTransformer] = None
        self.chunks:  List[Chunk]  = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self._built = False

        logger.info("RAGRetriever initialised (index not yet built — call build_index())")

    def _load_encoder(self):
        if self.encoder is None:
            logger.info(f"Loading embedding model: {self.embed_model_name}")
            self.encoder = SentenceTransformer(self.embed_model_name)
            logger.info("Embedding model ready.")

    def _chunk_text(self, text: str, doc_id: str,
                    doc_title: str, domain: str,
                    source: str) -> List[Chunk]:
        """Split document text into overlapping word-level chunks."""
        words  = text.split()
        chunks = []
        step   = max(1, self.chunk_size - self.chunk_overlap)

        for i, start in enumerate(range(0, len(words), step)):
            end        = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])

            if len(chunk_text.strip()) < 30:
                continue

            chunk_id = hashlib.md5(
                f"{doc_id}_{i}".encode()
            ).hexdigest()[:12]

            chunks.append(Chunk(
                chunk_id    = chunk_id,
                doc_id      = doc_id,
                doc_title   = doc_title,
                text        = chunk_text,
                domain      = domain,
                source      = source,
                chunk_index = i
            ))

            if end >= len(words):
                break

        return chunks

    def add_document(self, title: str, content: str,
                     domain: str = "general",
                     source: str = "custom",
                     doc_id: str = None) -> str:
        """
        Add a new document to the knowledge base.
        Call rebuild_index() after adding documents.

        Returns the doc_id.
        """
        if doc_id is None:
            doc_id = "doc_" + hashlib.md5(
                title.encode()
            ).hexdigest()[:8]

        new_chunks = self._chunk_text(
            content, doc_id, title, domain, source
        )
        self.chunks.extend(new_chunks)
        logger.info(f"Added document '{title}' → {len(new_chunks)} chunks "
                    f"(total: {len(self.chunks)})")
        self._built = False   # mark as stale
        return doc_id

    def build_index(self, force_rebuild: bool = False):
        """
        Embed all chunks and build the FAISS (or numpy) search index.
        Loads from disk if a saved index exists and force_rebuild=False.
        """
        index_pkl = self.index_path + f"_cs{self.chunk_size}.pkl"

        if not force_rebuild and os.path.exists(index_pkl):
            logger.info(f"Loading existing RAG index from {index_pkl}")
            self._load_from_disk(index_pkl)
            return

        # Start from built-in documents
        if not self.chunks:
            self._ingest_builtin_docs()

        self._load_encoder()

        logger.info(f"Embedding {len(self.chunks)} chunks ...")
        texts = [c.text for c in self.chunks]
        self.embeddings = self.encoder.encode(
            texts, convert_to_numpy=True,
            show_progress_bar=True, batch_size=32
        )

        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)   # inner product = cosine
            self.faiss_index.add(self.embeddings.astype(np.float32))
            logger.info(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")
        else:
            logger.info("Using numpy cosine search (install faiss-cpu for faster search).")

        self._built = True
        self._save_to_disk(index_pkl)

    def _ingest_builtin_docs(self):
        """Load the built-in knowledge base documents."""
        for doc in BUILTIN_DOCUMENTS:
            chunks = self._chunk_text(
                doc["content"], doc["doc_id"],
                doc["title"], doc["domain"], doc["source"]
            )
            self.chunks.extend(chunks)
        logger.info(f"Ingested {len(BUILTIN_DOCUMENTS)} built-in documents "
                    f"→ {len(self.chunks)} chunks total.")

    def _save_to_disk(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks":     self.chunks,
                "embeddings": self.embeddings,
                "model":      self.embed_model_name
            }, f)
        logger.info(f"RAG index saved to {path}")

    def _load_from_disk(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks     = data["chunks"]
        self.embeddings = data["embeddings"]

        if FAISS_AVAILABLE:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings.astype(np.float32))

        self._built = True
        logger.info(f"RAG index loaded: {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 3,
                 domain_filter: str = None) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query:         The question or claim to find context for
            top_k:         Number of chunks to return
            domain_filter: Optional domain to restrict search (e.g. "science")

        Returns:
            List of RetrievalResult sorted by relevance score descending
        """
        if not self._built:
            self.build_index()

        self._load_encoder()

        # Encode and normalise query
        q_emb = self.encoder.encode([query], convert_to_numpy=True)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # Apply domain filter by masking
        if domain_filter:
            valid_idx = [
                i for i, c in enumerate(self.chunks)
                if c.domain == domain_filter
            ]
            if not valid_idx:
                logger.warning(f"No chunks for domain '{domain_filter}', ignoring filter.")
                valid_idx = list(range(len(self.chunks)))
        else:
            valid_idx = list(range(len(self.chunks)))

        sub_embs = self.embeddings[valid_idx]

        # Search
        scores = (sub_embs @ q_emb.T).squeeze()   # cosine similarity

        # Get top-k indices
        k = min(top_k, len(valid_idx))
        top_local = np.argsort(scores)[::-1][:k]

        results = []
        for rank, local_i in enumerate(top_local):
            global_i = valid_idx[local_i]
            sim_score = float(scores[local_i])
            chunk     = self.chunks[global_i]
            logger.info(f"  Rank {rank+1}: '{chunk.doc_title}' "
                        f"[chunk {chunk.chunk_index}] score={sim_score:.4f}")
            results.append(RetrievalResult(
                chunk = chunk,
                score = sim_score,
                rank  = rank + 1
            ))

        return results

    def get_context_string(self, results: List[RetrievalResult],
                           max_chars: int = 2000,
                           min_score: float = 0.3) -> str:
        """
        Join retrieved chunks into a single context string for the scorers.
        Deduplicates by document and drops low-relevance chunks.
        """
        seen_docs = set()
        parts     = []
        total     = 0

        for r in results:
            # Drop chunks below minimum relevance threshold
            if r.score < min_score:
                logger.info(f"Skipping low-relevance chunk: '{r.chunk.doc_title}' "
                            f"(score={r.score:.3f} < {min_score})")
                continue

            chunk  = r.chunk
            header = f"[Source: {chunk.doc_title} ({chunk.source}), relevance={r.score:.3f}]"
            body   = chunk.text.strip()
            block  = f"{header}\n{body}"

            if total + len(block) > max_chars:
                break

            # Deduplicate: if we already have a chunk from this doc, skip
            if chunk.doc_id in seen_docs:
                continue
            seen_docs.add(chunk.doc_id)

            parts.append(block)
            total += len(block)

        return "\n\n".join(parts)

    def get_index_stats(self) -> Dict:
        """Return statistics about the current knowledge base."""
        domain_counts = {}
        source_counts = {}
        doc_ids       = set()

        for c in self.chunks:
            domain_counts[c.domain] = domain_counts.get(c.domain, 0) + 1
            source_counts[c.source] = source_counts.get(c.source, 0) + 1
            doc_ids.add(c.doc_id)

        return {
            "total_chunks":    len(self.chunks),
            "total_documents": len(doc_ids),
            "domains":         domain_counts,
            "sources":         source_counts,
            "index_built":     self._built,
            "embed_model":     self.embed_model_name,
            "faiss_available": FAISS_AVAILABLE
        }