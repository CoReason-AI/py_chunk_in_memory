# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

import pytest
from py_chunk_in_memory.preprocessors import (
    PreprocessorStep,
    WhitespaceNormalizer,
    UnicodeNormalizer,
)


class TestWhitespaceNormalizer:
    """Unit tests for the WhitespaceNormalizer preprocessor."""

    def test_initialization_invalid(self):
        """Test that initialization fails with invalid parameters."""
        with pytest.raises(
            ValueError, match="max_consecutive_newlines must be a non-negative integer."
        ):
            WhitespaceNormalizer(max_consecutive_newlines=-1)

    def test_empty_string(self):
        """Test that an empty string remains empty."""
        normalizer = WhitespaceNormalizer()
        assert normalizer.process("") == ""

    def test_string_with_only_whitespace(self):
        """Test that a string containing only whitespace becomes empty."""
        normalizer = WhitespaceNormalizer()
        assert normalizer.process("  \t\n\n  ") == ""

    def test_no_changes_needed(self):
        """Test a clean string that should not be modified."""
        text = "This is a clean string.\nWith a single newline."
        normalizer = WhitespaceNormalizer()
        assert normalizer.process(text) == text

    def test_trimming_leading_and_trailing_whitespace(self):
        """Test removal of leading/trailing spaces, tabs, and newlines."""
        text = "  \n\t  hello world \t\n "
        normalizer = WhitespaceNormalizer()
        assert normalizer.process(text) == "hello world"

    def test_collapse_multiple_spaces_and_tabs(self):
        """Test collapsing of multiple spaces and tabs into a single space."""
        text = "hello   world\t\tfrom\t a test"
        normalizer = WhitespaceNormalizer()
        assert normalizer.process(text) == "hello world from a test"

    def test_collapse_consecutive_newlines_default(self):
        """Test collapsing multiple newlines to a single one (default)."""
        text = "Line 1\n\n\nLine 2\n\nLine 3"
        normalizer = WhitespaceNormalizer()
        assert normalizer.process(text) == "Line 1\nLine 2\nLine 3"

    def test_collapse_consecutive_newlines_custom_max(self):
        """Test collapsing newlines to a custom maximum value."""
        text = "Paragraph 1\n\n\n\nParagraph 2"
        normalizer = WhitespaceNormalizer(max_consecutive_newlines=2)
        assert normalizer.process(text) == "Paragraph 1\n\nParagraph 2"

    def test_remove_all_newlines(self):
        """Test removing all newlines when max_consecutive_newlines is 0."""
        text = "First line.\nSecond line.\n\nThird line."
        normalizer = WhitespaceNormalizer(max_consecutive_newlines=0)
        assert normalizer.process(text) == "First line. Second line. Third line."

    def test_mixed_whitespace_scenario(self):
        """A complex test combining all normalization rules."""
        text = " \t\n Start of text.  With   extra spaces.\n\n\nAnd multiple\n\n\n\nnewlines. \t End. \n "
        normalizer = WhitespaceNormalizer(max_consecutive_newlines=2)
        expected = "Start of text. With extra spaces.\n\nAnd multiple\n\nnewlines. End."
        assert normalizer.process(text) == expected

    def test_no_newlines_with_max_gt_zero(self):
        """Test that text without newlines is unaffected by newline rule."""
        text = "  A sentence \t with   spaces.  "
        normalizer = WhitespaceNormalizer(max_consecutive_newlines=2)
        assert normalizer.process(text) == "A sentence with spaces."

    def test_newlines_equal_to_max(self):
        """Test that newlines matching the max are not collapsed."""
        text = "Line 1\n\nLine 2"
        normalizer = WhitespaceNormalizer(max_consecutive_newlines=2)
        assert normalizer.process(text) == "Line 1\n\nLine 2"


def test_preprocessor_step_abc_cannot_be_instantiated():
    """Verify that the abstract PreprocessorStep class cannot be instantiated."""
    with pytest.raises(TypeError):
        PreprocessorStep()


def test_preprocessor_step_abc_process_raises_not_implemented():
    """Verify that the process method of a minimal subclass raises NotImplementedError."""

    class MinimalPreprocessor(PreprocessorStep):
        def process(self, text: str) -> str:
            # The type ignore is necessary because mypy flags this as an unsafe
            # super() call on an abstract method, which is the exact thing
            # we want to test.
            return super().process(text)  # type: ignore [safe-super]

    preprocessor = MinimalPreprocessor()
    with pytest.raises(NotImplementedError):
        preprocessor.process("test")


class TestUnicodeNormalizer:
    """Unit tests for the UnicodeNormalizer preprocessor."""

    def test_initialization_invalid_form(self):
        """Test that initialization fails with an invalid normalization form."""
        with pytest.raises(
            ValueError, match="form must be one of 'NFC', 'NFKC', 'NFD', 'NFKD'"
        ):
            UnicodeNormalizer(form="INVALID")

    def test_initialization_default_form(self):
        """Test that the default normalization form is 'NFC'."""
        normalizer = UnicodeNormalizer()
        assert normalizer.form == "NFC"

    def test_empty_string(self):
        """Test that an empty string remains empty after processing."""
        normalizer = UnicodeNormalizer()
        assert normalizer.process("") == ""

    @pytest.mark.parametrize(
        "form, text, expected",
        [
            # NFC: Composed form
            ("NFC", "\u0065\u0301", "\u00e9"),  # e + ´ -> é
            ("NFC", "\u00e9", "\u00e9"),  # é -> é (already composed)
            # NFD: Decomposed form
            ("NFD", "\u00e9", "\u0065\u0301"),  # é -> e + ´
            (
                "NFD",
                "\u0065\u0301",
                "\u0065\u0301",
            ),  # e + ´ -> e + ´ (already decomposed)
            # NFKC: Compatibility composed
            ("NFKC", "\ufb01", "fi"),  # ﬁ -> fi
            # NFKD: Compatibility decomposed
            ("NFKD", "\ufb01", "fi"),  # ﬁ -> fi
            ("NFKD", "1\u2075", "15"),  # ¹⁵ -> 15
        ],
    )
    def test_normalization_forms(self, form: str, text: str, expected: str):
        """Test all four Unicode normalization forms."""
        normalizer = UnicodeNormalizer(form=form)
        assert normalizer.process(text) == expected

    def test_no_changes_needed(self):
        """Test that a standard ASCII string is not modified."""
        text = "This is a standard string."
        normalizer = UnicodeNormalizer(form="NFC")
        assert normalizer.process(text) == text
