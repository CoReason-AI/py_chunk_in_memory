# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from abc import ABC, abstractmethod


class PreprocessorStep(ABC):
    """Abstract base class for a single step in a preprocessing pipeline."""

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Processes the input text and returns the modified text.

        Args:
            text: The input text to process.

        Returns:
            The processed text.
        """
        raise NotImplementedError


class WhitespaceNormalizer(PreprocessorStep):
    """
    A preprocessor for cleaning and standardizing whitespace in a text.

    This step performs the following actions in order:
    1. Trims leading and trailing whitespace from the text.
    2. Replaces multiple consecutive spaces or tabs with a single space.
    3. Collapses multiple consecutive newline characters into a specified maximum.
    """

    def __init__(
        self,
        max_consecutive_newlines: int = 1,
    ):
        """
        Initializes the WhitespaceNormalizer.

        Args:
            max_consecutive_newlines: The maximum number of consecutive newline
                                      characters to allow. Defaults to 1.
        """
        if max_consecutive_newlines < 0:
            raise ValueError("max_consecutive_newlines must be a non-negative integer.")
        self.max_consecutive_newlines = max_consecutive_newlines

    def process(self, text: str) -> str:
        """
        Applies whitespace normalization to the text.
        """
        import re

        # 1. Trim leading/trailing whitespace
        processed_text = text.strip()

        # 2. Collapse multiple spaces/tabs into a single space
        processed_text = re.sub(r"[ \t]+", " ", processed_text)

        # 3. Collapse consecutive newlines
        if self.max_consecutive_newlines > 0:
            newline_pattern = r"\n{" + str(self.max_consecutive_newlines + 1) + r",}"
            replacement = "\n" * self.max_consecutive_newlines
            processed_text = re.sub(newline_pattern, replacement, processed_text)
        elif self.max_consecutive_newlines == 0:
            # If 0, remove all newlines
            processed_text = re.sub(r"\n+", " ", processed_text)

        return processed_text


class UnicodeNormalizer(PreprocessorStep):
    """
    A preprocessor for applying Unicode normalization to a text.

    This is useful for ensuring that text is in a consistent, canonical form,
    which is important for accurate tokenization and embedding.
    """

    def __init__(self, form: str = "NFC"):
        """
        Initializes the UnicodeNormalizer.

        Args:
            form: The normalization form to apply. One of 'NFC', 'NFKC',
                  'NFD', 'NFKD'. Defaults to 'NFC'.
        """
        if form not in ["NFC", "NFKC", "NFD", "NFKD"]:
            raise ValueError("form must be one of 'NFC', 'NFKC', 'NFD', 'NFKD'.")
        self.form = form

    def process(self, text: str) -> str:
        """
        Applies the specified Unicode normalization form to the text.
        """
        import unicodedata

        return unicodedata.normalize(self.form, text)  # type: ignore[arg-type]
