# llm_judge.py — LLM-as-Judge Hallucination Detector using Gemini API
# Uses the NEW google-genai SDK (google.genai), replacing deprecated google.generativeai

import json
import time
import logging
import re
from typing import Dict, List, Optional

from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class GeminiJudge:
    """
    Uses Gemini as an LLM judge to evaluate claim faithfulness.

    Implements two judge strategies:
    1. Direct scoring (single prompt)
    2. Chain-of-thought scoring (reason first, then score)
    """

    DIRECT_PROMPT = """You are an expert hallucination detector. Analyze whether the CLAIM is faithful to the CONTEXT.

CONTEXT:
{context}

CLAIM:
{claim}

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
  "verdict": "FAITHFUL" | "HALLUCINATED" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0>,
  "faithfulness_score": <float 0.0 to 1.0>,
  "reason": "<one concise sentence>",
  "claim_type": "factual" | "inferential" | "contradictory" | "fabricated" | "unsupported"
}}"""

    COT_PROMPT = """You are an expert fact-checker. Analyze whether the CLAIM is supported by the CONTEXT using chain-of-thought reasoning.

CONTEXT:
{context}

CLAIM:
{claim}

Step 1: Identify key facts in the claim.
Step 2: Check each fact against the context.
Step 3: Determine if any facts are contradicted or unsupported.
Step 4: Provide a faithfulness score.

After your analysis, output a JSON block like:
```json
{{
  "verdict": "FAITHFUL" | "HALLUCINATED" | "UNCERTAIN",
  "confidence": <float 0.0 to 1.0>,
  "faithfulness_score": <float 0.0 to 1.0>,
  "reason": "<one concise sentence>",
  "claim_type": "factual" | "inferential" | "contradictory" | "fabricated" | "unsupported",
  "key_facts_checked": ["<fact1>", "<fact2>"]
}}
```"""

    BATCH_PROMPT = """You are an expert hallucination detector. Score each CLAIM against the CONTEXT.

CONTEXT:
{context}

CLAIMS TO EVALUATE:
{claims_numbered}

For each claim, respond with ONLY a JSON array (no markdown):
[
  {{"claim_id": 1, "verdict": "FAITHFUL"|"HALLUCINATED"|"UNCERTAIN", "faithfulness_score": <0.0-1.0>, "reason": "<brief>"}},
  ...
]"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.client     = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.call_count = 0
        logger.info(f"GeminiJudge initialized with model: {model_name} (google-genai SDK)")

    def _call_gemini(self, prompt: str, max_retries: int = 4) -> Optional[str]:
        """Call Gemini API with rate-limit-aware retry logic."""
        import re as _re
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        system_instruction=(
                            "You are a hallucination detection system. "
                            "Always respond with ONLY valid JSON. "
                            "Never include markdown fences, preamble, or explanation outside the JSON object. "
                            "Your entire response must be parseable by json.loads()."
                        ),
                    )
                )
                self.call_count += 1
                return response.text
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini API attempt {attempt+1} failed: {e}")

                if attempt >= max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed.")
                    return None

                # Parse suggested retry delay from error message if present
                delay_match = _re.search(r"retry[^0-9]*([0-9]+(?:\.[0-9]+)?)s", err_str)
                if delay_match:
                    wait = min(float(delay_match.group(1)), 65)  # cap at 65s
                    logger.info(f"Rate limited — waiting {wait:.0f}s as suggested by API...")
                    time.sleep(wait)
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = 60  # default 60s for rate limit
                    logger.info(f"Rate limited — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    time.sleep(2 ** attempt)

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Robustly parse JSON from LLM response.
        
        Handles: plain JSON, markdown fences, thinking preamble,
        and truncated responses from high-reasoning models like Gemini 2.5.
        """
        if not text:
            return None

        # Strip <think>...</think> blocks (Gemini 2.5 / o1-style thinking)
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

        # 1. Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # 2. Extract from ```json ... ``` or ``` ... ``` fences
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 3. Find the first complete {...} JSON object (handles preamble text)
        # Walk from the first { to find a balanced closing }
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break

        # 4. Fallback: regex for any {...} blob
        obj_match = re.search(r"\{[\s\S]*?\}", text)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from: {text[:300]}")
        return None

    def score_claim(self, context: str, claim: str, use_cot: bool = False) -> Dict:
        prompt_template = self.COT_PROMPT if use_cot else self.DIRECT_PROMPT
        prompt = prompt_template.format(context=context, claim=claim)
        raw_response = self._call_gemini(prompt)
        if raw_response is None:
            return self._default_score(context, claim, error="API_FAILED")
        parsed = self._parse_json_response(raw_response)
        if parsed is None:
            return self._default_score(context, claim, error="PARSE_FAILED")
        result = {
            "verdict":            parsed.get("verdict", "UNCERTAIN"),
            "confidence":         float(parsed.get("confidence", 0.5)),
            "faithfulness_score": float(parsed.get("faithfulness_score", 0.5)),
            "reason":             parsed.get("reason", ""),
            "claim_type":         parsed.get("claim_type", "unknown"),
            "context":            context[:200],
            "claim":              claim,
            "model":              self.model_name,
            "used_cot":           use_cot,
            "raw_response":       raw_response[:500] if raw_response else ""
        }
        if use_cot and "key_facts_checked" in parsed:
            result["key_facts_checked"] = parsed["key_facts_checked"]
        return result
    def score_document(self, context: str, generated_text: str, use_cot: bool = False) -> Dict:
        import nltk
        try:
            sentences = nltk.sent_tokenize(generated_text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(generated_text)

        sentences = [s for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return {
                "faithfulness_score": 1.0,
                "sentence_scores": [],
                "verdict": "UNCERTAIN",
                "confidence": 0.5,
                "hallucination_rate": 0.0,
                "total_sentences": 0
            }

        # Single Gemini batch call instead of one call per sentence
        sentence_scores = self.batch_score_efficient(context, sentences)

        for i, sent in enumerate(sentences):
           sentence_scores[i]["sentence"] = sent

        import numpy as np
        fs_values = [s["faithfulness_score"] for s in sentence_scores]
        verdicts = [s["verdict"] for s in sentence_scores]

        hallucinated = verdicts.count("HALLUCINATED")
        faithful = verdicts.count("FAITHFUL")
        uncertain = verdicts.count("UNCERTAIN")

        overall = max(
            ["FAITHFUL", "HALLUCINATED", "UNCERTAIN"],
            key=lambda v: verdicts.count(v)
        )

        return {
            "faithfulness_score": float(np.mean(fs_values)),
            "verdict": overall,
            "confidence": float(np.mean([
                s.get("confidence", 0.5) for s in sentence_scores
            ])),
            "hallucinated_sentences": hallucinated,
            "faithful_sentences": faithful,
            "uncertain_sentences": uncertain,
            "total_sentences": len(sentence_scores),
            "hallucination_rate": hallucinated / len(sentence_scores),
            "sentence_scores": sentence_scores
        }



    def batch_score_efficient(self, context: str, claims: List[str]) -> List[Dict]:
        claims_numbered = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        prompt = self.BATCH_PROMPT.format(context=context, claims_numbered=claims_numbered)
        raw_response = self._call_gemini(prompt)
        if not raw_response:
            return [self._default_score(context, c) for c in claims]
        try:
            arr_match = re.search(r"\[[\s\S]*\]", raw_response)
            if arr_match:
                parsed_list = json.loads(arr_match.group())
                return [{
                    "claim":              claims[i] if i < len(claims) else "",
                    "context":            context[:200],
                    "verdict":            item.get("verdict", "UNCERTAIN"),
                    "faithfulness_score": float(item.get("faithfulness_score", 0.5)),
                    "reason":             item.get("reason", ""),
                    "model":              self.model_name
                } for i, item in enumerate(parsed_list)]
        except Exception as e:
            logger.warning(f"Batch parse failed: {e}")
        return [self.score_claim(context, c) for c in claims]

    @staticmethod
    def _default_score(context: str, claim: str, error: str = "UNKNOWN") -> Dict:
        return {
            "verdict": "UNCERTAIN", "confidence": 0.5,
            "faithfulness_score": 0.5,
            "reason": f"Scoring failed: {error}",
            "claim_type": "unknown", "context": context[:200],
            "claim": claim, "error": error
        }

    def get_usage_stats(self) -> Dict:
        return {"api_calls": self.call_count, "model": self.model_name}