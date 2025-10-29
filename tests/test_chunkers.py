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
from py_chunk_in_memory.chunkers import BaseChunker, FixedSizeChunker, RecursiveCharacterChunker


def test_base_chunker_can_be_instantiated_with_default_len_func():
    """Verify BaseChunker can be instantiated and has a default length_function."""

    # This is a concrete implementation for testing purposes
    class ConcreteChunker(BaseChunker):
        def chunk(self, text: str, **kwargs):
            yield from []  # pragma: no cover

    chunker = ConcreteChunker()
    assert chunker._length_function("hello") == 5


def test_base_chunker_chunk_raises_not_implemented():
    """Verify that calling chunk on a minimal subclass raises NotImplementedError."""

    class MinimalChunker(BaseChunker):
        def chunk(self, text: str, **kwargs):
            return super().chunk(text, **kwargs)  # type: ignore [safe-super]

    chunker = MinimalChunker()
    with pytest.raises(NotImplementedError):
        chunker.chunk("test")


def test_fixed_size_chunker_basic():
    """Test basic fixed-size chunking with default len function and no overlap."""
    text = "This is a test string for basic chunking."
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 5
    assert chunks[0].text_for_generation == "This is a "
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 10
    assert chunks[0].sequence_number == 0

    assert chunks[1].text_for_generation == "test strin"
    assert chunks[1].start_char_index == 10
    assert chunks[1].end_char_index == 20
    assert chunks[1].sequence_number == 1

    assert chunks[2].text_for_generation == "g for basi"
    assert chunks[2].start_char_index == 20
    assert chunks[2].end_char_index == 30
    assert chunks[2].sequence_number == 2

    assert chunks[3].text_for_generation == "c chunking"
    assert chunks[3].start_char_index == 30
    assert chunks[3].end_char_index == 40
    assert chunks[3].sequence_number == 3

    assert chunks[4].text_for_generation == "."
    assert chunks[4].start_char_index == 40
    assert chunks[4].end_char_index == 41
    assert chunks[4].sequence_number == 4


def test_fixed_size_chunker_with_overlap():
    """Test fixed-size chunking with overlap and default len function."""
    text = "abcdefghijklmnopqrstuvwxyz"
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 4
    assert chunks[0].text_for_generation == "abcdefghij"
    assert chunks[1].text_for_generation == "hijklmnopq"
    assert chunks[2].text_for_generation == "opqrstuvwx"
    assert chunks[3].text_for_generation == "vwxyz"

    assert chunks[0].start_char_index == 0
    assert chunks[1].start_char_index == 7  # 10 - 3
    assert chunks[2].start_char_index == 14  # 7 + (10 - 3)
    assert chunks[3].start_char_index == 21  # 14 + (10-3)


def test_fixed_size_chunker_edge_case_empty_string():
    """Test chunking an empty string."""
    text = ""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 0


def test_fixed_size_chunker_edge_case_small_string():
    """Test chunking a string smaller than the chunk size."""
    text = "short"
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "short"
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 5


def test_fixed_size_chunker_invalid_params():
    """Test that the chunker raises errors for invalid parameters."""
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, chunk_overlap=0)  # zero chunk size
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=-10, chunk_overlap=0)  # negative chunk size
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=-1)  # negative overlap
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=10)  # overlap equals chunk size
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=11)  # overlap > chunk size


def test_chunk_metadata_is_correct():
    """Verify that metadata fields are correctly populated with default len."""
    text = "Testing metadata population."
    chunker = FixedSizeChunker(chunk_size=8, chunk_overlap=2)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 5
    for i, chunk in enumerate(chunks):
        assert chunk.sequence_number == i
        assert chunk.chunking_strategy_used == "fixed_size"
        expected_start = i * (8 - 2)
        assert chunk.start_char_index == expected_start
        expected_end = min(expected_start + 8, len(text))
        assert chunk.end_char_index == expected_end
        # Also check token_count which should match char count here
        assert chunk.token_count == len(chunk.text_for_generation)


def test_fixed_size_chunker_with_unicode_characters():
    """Test chunking with Unicode characters and default len."""
    text = "こんにちは、世界！"  # "Hello, world!" in Japanese
    chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=1)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "こんにちは"
    assert chunks[1].text_for_generation == "は、世界！"


def test_fixed_size_chunker_perfect_fit():
    """Test chunking where text length is a perfect multiple of chunk size."""
    text = "1234567890"
    chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "12345"
    assert chunks[1].text_for_generation == "67890"


def test_fixed_size_chunker_whitespace_only():
    """Test chunking with text containing only whitespace."""
    text = "          "  # 10 spaces
    chunker = FixedSizeChunker(chunk_size=3, chunk_overlap=1)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 5
    assert chunks[0].text_for_generation == "   "
    assert chunks[1].text_for_generation == "   "
    assert chunks[2].text_for_generation == "   "
    assert chunks[3].text_for_generation == "   "
    assert chunks[4].text_for_generation == "  "


def test_fixed_size_chunker_with_extra_kwargs():
    """Test that the chunker handles unexpected kwargs gracefully."""
    text = "This is a test."
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    # The `unused_param` should be ignored
    chunks = list(chunker.chunk(text, unused_param="some_value"))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "This is a "
    assert chunks[1].text_for_generation == "test."


def test_fixed_size_chunker_with_custom_length_function():
    """Verify the chunker respects a custom word-counting length function."""

    # This function counts words (space-separated strings). It is not perfect
    # because it will count empty strings from multiple spaces, but it's
    # suitable for this test's purpose.
    def word_counter(text: str) -> int:
        return len(text.split(" "))

    text = "one two three four five six seven eight nine ten eleven"
    # Chunk into segments of 3 words, with 1 word overlap
    chunker = FixedSizeChunker(
        chunk_size=3, chunk_overlap=1, length_function=word_counter
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 5
    assert chunks[0].text_for_generation == "one two three"
    assert word_counter(chunks[0].text_for_generation) == 3
    assert chunks[1].text_for_generation == "three four five"
    assert word_counter(chunks[1].text_for_generation) == 3
    assert chunks[2].text_for_generation == "five six seven"
    assert word_counter(chunks[2].text_for_generation) == 3
    assert chunks[3].text_for_generation == "seven eight nine"
    assert word_counter(chunks[3].text_for_generation) == 3
    assert chunks[4].text_for_generation == "nine ten eleven"
    assert word_counter(chunks[4].text_for_generation) == 3

    # Check character indices to ensure they are being tracked correctly
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 13

    # Overlap should start at "three"
    assert chunks[1].start_char_index == 8
    assert chunks[1].end_char_index == 23


def test_chunker_handles_single_char_exceeding_chunk_size():
    """Test that a single character exceeding chunk size is handled."""

    # A length function where 'X' has a size of 100
    def length_function(x: str) -> int:
        return 100 if "X" in x else len(x)

    text = "abcXdef"
    chunker = FixedSizeChunker(
        chunk_size=10, chunk_overlap=2, length_function=length_function
    )
    chunks = list(chunker.chunk(text))

    # The logic should create three chunks: "abc", "X", and "def"
    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "abc"
    assert chunks[1].text_for_generation == "X"
    assert chunks[2].text_for_generation == "def"


def test_recursive_chunker_basic_no_overlap():
    """Test basic merging of splits with no overlap."""
    text = "Hi there. My name is Jules. What is your name?"
    chunker = RecursiveCharacterChunker(
        chunk_size=30, chunk_overlap=0, separators=[". "], keep_separator=True
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "Hi there. My name is Jules. "
    assert chunks[1].text_for_generation == "What is your name?"


def test_recursive_chunker_with_overlap():
    """Test recursive chunking with a realistic overlap scenario."""
    text = "a b c d e f g h i j k"
    chunker = RecursiveCharacterChunker(
        chunk_size=10, chunk_overlap=4, separators=[" "], keep_separator=True
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 3
    # As determined by manual trace:
    # Chunk 1: 'a b c d e ' (len 10)
    # Overlap: 'd e ' (len 4)
    # Chunk 2: 'd e f g h ' (len 10)
    # Overlap: 'g h ' (len 4)
    # Chunk 3: 'g h i j k' (len 9)
    assert chunks[0].text_for_generation == "a b c d e "
    assert chunks[1].text_for_generation == "d e f g h "
    assert chunks[2].text_for_generation == "g h i j k"


def test_recursive_chunker_long_word_fallback():
    """Test that a word longer than chunk_size is handled by the fallback."""
    text = "ThisIsAVeryLongWordWithoutAnySeparators"
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0, keep_separator=True)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 4
    assert chunks[0].text_for_generation == "ThisIsAVer"
    assert chunks[1].text_for_generation == "yLongWordW"
    assert chunks[2].text_for_generation == "ithoutAnyS"
    assert chunks[3].text_for_generation == "eparators"


def test_recursive_chunker_custom_separators_and_no_keep():
    """Test custom separators where the separator is discarded."""
    text = "one--two--three-four-five"
    chunker = RecursiveCharacterChunker(
        chunk_size=12, chunk_overlap=0, separators=["--", "-"], keep_separator=False
    )
    chunks = list(chunker.chunk(text))

    # With chunk_size=12, "onetwothree" (len 11) fits, but adding "four" does not.
    # The next chunk starts with "four" and adds "five".
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "onetwothree"
    assert chunks[1].text_for_generation == "fourfive"


def test_recursive_chunker_empty_string():
    """Test that an empty string produces no chunks."""
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
    chunks = list(chunker.chunk(text=""))
    assert len(chunks) == 0


def test_recursive_chunker_small_string():
    """Test a string smaller than the chunk size."""
    text = "This is small."
    chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=0)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "This is small."
