"""Text summarization module."""

import logging
import re

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Handles text summarization (abstractive or extractive)"""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self.model = None
        if model_name:
            from transformers import pipeline

            logger.info(f"Loading summarization model: {model_name}")
            self.model = pipeline(
                "summarization",
                model=model_name,
                device=0 if device == "cuda" else -1,
            )

    def summarize(self, text: str, title: str = "", max_sentences: int = 3) -> str:
        """
        Summarize text using abstractive or extractive methods

        Args:
            text: Text to summarize
            title: Optional title to include in summary
            max_sentences: Maximum sentences for extractive summary

        Returns:
            Summary text
        """
        if not text:
            return title or "No content"

        # Truncate very long content
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length]

        if self.model:
            # Limit input length for transformer models
            max_input = 1024
            text_truncated = text[:max_input]

            result = self.model(
                text_truncated, max_length=130, min_length=30, do_sample=False
            )
            return result[0]["summary_text"]

        # Fallback: Extractive summarization
        return self._extractive_summary(text, title, max_sentences)

    def _extractive_summary(
        self, text: str, title: str = "", max_sentences: int = 3
    ) -> str:
        """Simple extractive summarization by selecting first sentences"""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return title or text[:200]

        summary_parts = []
        if title:
            summary_parts.append(title)

        num_sentences = min(max_sentences, len(sentences))
        summary_parts.extend(sentences[:num_sentences])

        summary = ". ".join(summary_parts)
        if not summary.endswith("."):
            summary += "."

        return summary
