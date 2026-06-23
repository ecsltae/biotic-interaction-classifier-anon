"""
LLM Validator for Training Data Quality Control (Gate 6)

Uses an LLM to validate training data by checking if:
- Positives actually describe biotic interactions
- Negatives do NOT describe biotic interactions

The LLM acts as a secondary filter - disagreements are removed from training.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMValidator:
    """
    Validates training data using LLM as a quality filter.

    Supports:
    - Anthropic API (via anthropic package)
    - MCP (if configured)
    - Batch processing for efficiency
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-3-haiku-20240307",  # Use Haiku for speed/cost
        batch_size: int = 10
    ):
        """
        Initialize LLM validator.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            model: Model to use for validation
            batch_size: Number of sentences to validate per API call
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.batch_size = batch_size
        self.client = None

        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client with model: {model}")
            except ImportError:
                logger.warning("anthropic package not installed. Run: pip install anthropic")

    def _create_prompt(self, sentences: List[Tuple[int, str, int]]) -> str:
        """
        Create a batch validation prompt.

        Args:
            sentences: List of (idx, text, label) tuples

        Returns:
            Formatted prompt string
        """
        prompt = """You are validating training data for a biotic interaction classifier.

For each sentence, determine if it describes a BIOTIC INTERACTION between organisms.

Biotic interactions include:
- Predation (animal preys on/eats another)
- Parasitism (parasite infects/lives on host)
- Pollination (pollinator visits/pollinates plant)
- Herbivory (animal eats plant)
- Symbiosis/Mutualism
- Pathogenic infection
- Vector transmission (vector carries pathogen to host)
- Host relationships

NOT biotic interactions:
- Simple co-occurrence (two species found together without interaction)
- Taxonomic comparisons (species A is related to species B)
- Methodology descriptions (we collected species A and B)
- Single species descriptions
- Hypothetical/potential interactions without actual occurrence
- In vitro lab tests (antibacterial activity against)

For each sentence, respond with:
- "YES" if it describes an actual biotic interaction
- "NO" if it does NOT describe an actual biotic interaction

Respond in JSON format: {"results": [{"idx": N, "answer": "YES/NO"}, ...]}

Sentences to validate:
"""
        for idx, text, label in sentences:
            prompt += f"\n{idx}. {text}"

        return prompt

    def validate_sentence(self, sentence: str) -> Dict:
        """
        Validate a single sentence.

        Args:
            sentence: Text to validate

        Returns:
            {"is_interaction": bool, "confidence": float}
        """
        if not self.client:
            logger.warning("No LLM client available, skipping validation")
            return {"is_interaction": True, "confidence": 0.0}

        prompt = f"""Determine if this sentence describes a biotic interaction between organisms.

Sentence: "{sentence}"

A biotic interaction means one organism directly affects another (predation, parasitism,
pollination, infection, host relationship, etc.).

NOT a biotic interaction: co-occurrence, taxonomic comparison, methodology, single species.

Answer with just "YES" or "NO"."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.content[0].text.strip().upper()
            is_interaction = answer.startswith("YES")
            return {"is_interaction": is_interaction, "confidence": 0.9}
        except Exception as e:
            logger.error(f"LLM validation error: {e}")
            return {"is_interaction": True, "confidence": 0.0}

    def validate_batch(self, sentences: List[Tuple[int, str, int]]) -> Dict[int, bool]:
        """
        Validate a batch of sentences.

        Args:
            sentences: List of (idx, text, label) tuples

        Returns:
            Dict mapping idx -> is_interaction
        """
        if not self.client:
            logger.warning("No LLM client available, skipping batch validation")
            return {idx: True for idx, _, _ in sentences}

        prompt = self._create_prompt(sentences)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip()

            # Parse JSON response
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                results = {}
                for item in data.get("results", []):
                    idx = item.get("idx")
                    answer = item.get("answer", "").upper()
                    results[idx] = answer.startswith("YES")
                return results
            else:
                logger.warning(f"Could not parse LLM response: {response_text[:100]}")
                return {idx: True for idx, _, _ in sentences}

        except Exception as e:
            logger.error(f"LLM batch validation error: {e}")
            return {idx: True for idx, _, _ in sentences}

    def clean_dataset(
        self,
        df: pd.DataFrame,
        sample_size: int = None,
        progress_callback=None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate ALL sentences and REMOVE disagreements.

        - Positives where LLM says NO -> removed
        - Negatives where LLM says YES -> removed

        Args:
            df: Training data DataFrame with 'text' and 'label' columns
            sample_size: If set, only validate a sample (for testing)
            progress_callback: Optional callback for progress updates

        Returns:
            (cleaned_df, stats_dict)
        """
        if not self.client:
            logger.warning("No LLM client available, returning original dataset")
            return df, {"status": "skipped", "reason": "no_llm_client"}

        # Prepare data for validation
        data = list(df[['text', 'label']].itertuples())
        if sample_size:
            import random
            data = random.sample(data, min(sample_size, len(data)))

        total = len(data)
        logger.info(f"Validating {total} sentences with LLM...")

        # Process in batches
        indices_to_remove = set()
        removed_positives = 0
        removed_negatives = 0

        for i in range(0, total, self.batch_size):
            batch = data[i:i + self.batch_size]
            sentences = [(row.Index, row.text, row.label) for row in batch]

            results = self.validate_batch(sentences)

            for idx, text, label in sentences:
                llm_says_interaction = results.get(idx, True)
                is_positive = label == 1

                # Remove if LLM disagrees with label
                if is_positive and not llm_says_interaction:
                    indices_to_remove.add(idx)
                    removed_positives += 1
                elif not is_positive and llm_says_interaction:
                    indices_to_remove.add(idx)
                    removed_negatives += 1

            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)

            if (i + self.batch_size) % 100 == 0:
                logger.info(f"  Processed {min(i + self.batch_size, total)}/{total}")

        # Create cleaned DataFrame
        cleaned_df = df.drop(index=list(indices_to_remove))

        stats = {
            "total_validated": total,
            "removed_positives": removed_positives,
            "removed_negatives": removed_negatives,
            "total_removed": len(indices_to_remove),
            "original_size": len(df),
            "cleaned_size": len(cleaned_df)
        }

        logger.info(f"LLM validation complete:")
        logger.info(f"  Removed {removed_positives} positives (LLM said NO)")
        logger.info(f"  Removed {removed_negatives} negatives (LLM said YES)")
        logger.info(f"  Original: {len(df)} -> Cleaned: {len(cleaned_df)}")

        return cleaned_df, stats


def validate_with_llm(
    input_path: str,
    output_path: str = None,
    sample_size: int = None,
    api_key: str = None
):
    """
    Convenience function to validate and clean a training data file.

    Args:
        input_path: Path to training data CSV
        output_path: Path to save cleaned data (optional)
        sample_size: Only validate a sample (for testing)
        api_key: Anthropic API key
    """
    # Load data
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    # Initialize validator
    validator = LLMValidator(api_key=api_key)

    # Clean dataset
    cleaned_df, stats = validator.clean_dataset(df, sample_size=sample_size)

    # Save if output path provided
    if output_path:
        cleaned_df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")

    return cleaned_df, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based training data validation")
    parser.add_argument("input_file", help="Path to training data CSV")
    parser.add_argument("--output", "-o", help="Output path for cleaned data")
    parser.add_argument("--sample", type=int, help="Only validate N samples (for testing)")
    parser.add_argument("--api-key", help="Anthropic API key")

    args = parser.parse_args()

    cleaned_df, stats = validate_with_llm(
        args.input_file,
        args.output,
        args.sample,
        args.api_key
    )

    print("\n=== LLM Validation Stats ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
