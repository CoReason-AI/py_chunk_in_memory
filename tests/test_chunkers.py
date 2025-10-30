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
from py_chunk_in_memory.models import Chunk
from py_chunk_in_memory.chunkers import (
    BaseChunker,
    FixedSizeChunker,
    RecursiveCharacterChunker,
)


class DummyChunker(BaseChunker):
    """A chunker that returns a predefined list of chunks for testing."""

    def __init__(self, chunks_to_return, chunk_size, **kwargs):
        super().__init__(**kwargs)
        self.chunks_to_return = chunks_to_return
        self.chunk_size = chunk_size

    def chunk(self, text: str, **kwargs):
        # We ignore the text and just return our pre-canned chunks
        return self._link_chunks(self.chunks_to_return, self.chunk_size)


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


def test_base_chunker_invalid_runt_handling_params():
    """Verify BaseChunker constructor raises errors for invalid runt handling params."""
    with pytest.raises(ValueError, match="minimum_chunk_size must be a non-negative integer."):
        FixedSizeChunker(chunk_size=10, minimum_chunk_size=-1)

    with pytest.raises(ValueError, match="runt_handling must be one of 'keep', 'discard', 'merge'."):
        FixedSizeChunker(chunk_size=10, runt_handling="invalid_policy")


def test_runt_handling_keep_policy():
    """Test the 'keep' policy for runt handling (default behavior)."""
    text = "This is a test. Runt."
    # The last chunk " Runt." has length 6. The final chunk "." has length 1.
    chunker = FixedSizeChunker(chunk_size=10, minimum_chunk_size=5)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "This is a "
    assert chunks[1].text_for_generation == "test. Runt"
    assert chunks[2].text_for_generation == "."
    assert len(chunks[2].text_for_generation) < 5 # This is a runt


def test_runt_handling_discard_policy():
    """Test the 'discard' policy for runt handling."""
    text = "This is a test. Runt."
    # Last chunk will be "." (a runt of size 1)
    chunker = FixedSizeChunker(chunk_size=10, minimum_chunk_size=5, runt_handling="discard")
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "This is a "
    assert chunks[1].text_for_generation == "test. Runt"


def test_runt_handling_merge_policy():
    """Test the 'merge' policy directly using the DummyChunker."""
    pre_canned_chunks = [
        Chunk(text_for_generation="This is the first chunk.", start_char_index=0, end_char_index=24),
        Chunk(text_for_generation="Runt", start_char_index=25, end_char_index=29),
    ]
    chunker = DummyChunker(
        pre_canned_chunks,
        chunk_size=30,
        minimum_chunk_size=10,
        runt_handling="merge"
    )
    chunks = chunker.chunk("doesn't matter")

    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "This is the first chunk.Runt"
    assert chunks[0].end_char_index == 29 # Verify end index is updated


def test_runt_handling_merge_does_not_exceed_chunk_size():
    """Test that merge policy does not merge if it would exceed original chunk size."""
    pre_canned_chunks = [
        Chunk(text_for_generation="1234567890", start_char_index=0, end_char_index=10),
        Chunk(text_for_generation="Runt", start_char_index=11, end_char_index=15),
    ]
    # chunk_size is 10, merged size would be 14. Merge should be rejected.
    chunker = DummyChunker(
        pre_canned_chunks,
        chunk_size=10,
        minimum_chunk_size=5,
        runt_handling="merge"
    )
    chunks = chunker.chunk("doesn't matter")

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "1234567890"
    assert chunks[1].text_for_generation == "Runt"


def test_runt_handling_merge_first_chunk_is_runt():
    """Test merge policy when the first chunk is a runt (it should be kept)."""
    pre_canned_chunks = [
        Chunk(text_for_generation="Runt", start_char_index=0, end_char_index=4),
        Chunk(text_for_generation="This is the second chunk.", start_char_index=5, end_char_index=29),
    ]
    chunker = DummyChunker(
        pre_canned_chunks,
        chunk_size=30,
        minimum_chunk_size=10,
        runt_handling="merge"
    )
    chunks = chunker.chunk("doesn't matter")

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "Runt" # Kept because it has no predecessor
    assert chunks[1].text_for_generation == "This is the second chunk."


def test_runt_handling_merge_consecutive_runts():
    """Test merge policy with multiple consecutive runts."""
    pre_canned_chunks = [
        Chunk(text_for_generation="A long sentence.", start_char_index=0, end_char_index=16),
        Chunk(text_for_generation="Runt1.", start_char_index=17, end_char_index=23),
        Chunk(text_for_generation="Runt2.", start_char_index=24, end_char_index=30),
    ]
    # Runt2 merges into Runt1. The result ("Runt1.Runt2.") is still a runt.
    # This new runt then merges into the first sentence.
    chunker = DummyChunker(
        pre_canned_chunks,
        chunk_size=35, # Large enough to allow all merges
        minimum_chunk_size=15,
        runt_handling="merge"
    )
    chunks = chunker.chunk("doesn't matter")

    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "A long sentence.Runt1.Runt2."
    assert chunks[0].end_char_index == 30


def test_runt_handling_updates_linking_and_sequence():
    """Verify chunk IDs and sequence numbers are correct after runt handling."""
    pre_canned_chunks = [
        Chunk(text_for_generation="Chunk1"),
        Chunk(text_for_generation="Runt1"),
        Chunk(text_for_generation="Chunk2"),
        Chunk(text_for_generation="Runt2"),
    ]
    # Runt1 merges into Chunk1, Runt2 merges into Chunk2
    chunker = DummyChunker(
        pre_canned_chunks,
        chunk_size=20,
        minimum_chunk_size=6,
        runt_handling="merge"
    )
    chunks = chunker.chunk("doesn't matter")

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "Chunk1Runt1"
    assert chunks[1].text_for_generation == "Chunk2Runt2"

    # Check sequence numbers
    assert chunks[0].sequence_number == 0
    assert chunks[1].sequence_number == 1

    # Check linking
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id is None


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
    chunker = RecursiveCharacterChunker(
        chunk_size=10, chunk_overlap=0, keep_separator=True
    )
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


def test_recursive_chunker_invalid_constructor_args():
    """Test that the RecursiveCharacterChunker constructor validates its arguments."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer."):
        RecursiveCharacterChunker(chunk_size=0)

    with pytest.raises(
        ValueError, match="chunk_overlap must be a non-negative integer."
    ):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=-1)

    with pytest.raises(
        ValueError, match="chunk_overlap must be smaller than chunk_size."
    ):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=10)


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


def test_recursive_chunker_re_error_fallback():
    """
    Test that a regex error during splitting falls back to character-wise split.
    """
    # The separator "(" is an invalid regex, causing a re.error
    chunker = RecursiveCharacterChunker(chunk_size=2, separators=["("])
    text = "a(b"
    chunks = list(chunker.chunk(text))
    # Should fall back to splitting by char, then merge them
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "a("
    assert chunks[1].text_for_generation == "b"


def test_fixed_size_chunker_avoids_empty_chunk():
    """Test that the chunker avoids creating an empty chunk."""

    def length_function(text: str) -> int:
        if text == "a":
            return 5
        if text == "b":
            return 5
        if text == "ab":
            return 11
        return len(text)

    text = "ab"
    chunker = FixedSizeChunker(
        chunk_size=10, chunk_overlap=0, length_function=length_function
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "a"
    assert chunks[1].text_for_generation == "b"


def test_chunk_linking():
    """Verify that previous_chunk_id and next_chunk_id are set correctly."""
    text = "Chunk one. Chunk two. Chunk three."
    chunker = FixedSizeChunker(chunk_size=12, chunk_overlap=2)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 4

    # First chunk
    assert chunks[0].previous_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id

    # Second chunk
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id == chunks[2].chunk_id

    # Third chunk
    assert chunks[2].previous_chunk_id == chunks[1].chunk_id
    assert chunks[2].next_chunk_id == chunks[3].chunk_id

    # Fourth chunk
    assert chunks[3].previous_chunk_id == chunks[2].chunk_id
    assert chunks[3].next_chunk_id is None


def test_recursive_character_chunker_linking():
    """Verify linking for the RecursiveCharacterChunker."""
    text = "Sentence one. Sentence two. Sentence three."
    chunker = RecursiveCharacterChunker(
        chunk_size=15, chunk_overlap=5, separators=[". "]
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 3
    assert chunks[0].previous_chunk_id is None
    assert chunks[0].next_chunk_id == chunks[1].chunk_id
    assert chunks[1].previous_chunk_id == chunks[0].chunk_id
    assert chunks[1].next_chunk_id == chunks[2].chunk_id
    assert chunks[2].previous_chunk_id == chunks[1].chunk_id
    assert chunks[2].next_chunk_id is None


def test_fixed_size_chunker_overlap_edge_case():
    """
    Test a specific edge case in the overlap calculation that was previously missed.
    This test ensures that when the overlap search results in a start position
    that is less than or equal to the current start_char, the chunker advances
    to the end of the current chunk.
    """

    # A length function where each character is size 3.
    # So "abc" has length 9. "de" has length 6.
    def length_function(text: str) -> int:
        return len(text) * 3

    text = "abcde"
    # chunk_size = 10, chunk_overlap = 2
    # 1. First chunk will be "abc" (length 9). end_char = 3.
    # 2. Overlap search:
    #    - start_of_overlap = 3
    #    - text[2:3] = "c", length = 3. This is > chunk_overlap, so loop breaks.
    #    - start_of_overlap remains 3.
    #    - This means the new start_char will be 3, "de"
    chunker = FixedSizeChunker(
        chunk_size=10, chunk_overlap=2, length_function=length_function
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "abc"
    assert chunks[1].text_for_generation == "de"


def test_recursive_chunker_tracks_char_indices():
    """Verify that RecursiveCharacterChunker correctly tracks character indices."""
    text = "First sentence. Second sentence. Third sentence."
    #        01234567890123456789012345678901234567890123456789
    #        0         1         2         3         4
    chunker = RecursiveCharacterChunker(
        chunk_size=20, chunk_overlap=5, separators=[". "], keep_separator=True
    )
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "First sentence. "
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 16

    assert chunks[1].text_for_generation == "Second sentence. "
    assert chunks[1].start_char_index == 16
    assert chunks[1].end_char_index == 33

    assert chunks[2].text_for_generation == "Third sentence."
    assert chunks[2].start_char_index == 33
    assert chunks[2].end_char_index == 48


def test_recursive_chunker_handles_final_split():
    """
    Tests that the final set of splits is correctly merged into a chunk.
    This covers the `if current_chunk_parts:` block at the end of
    `_merge_splits_with_indices`.
    """
    # This test simulates a scenario where the input text is split into parts,
    # and the final part needs to be processed correctly.
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
    text = "a b c"  # This will be split into ["a ", "b ", "c"]
    chunks = list(chunker.chunk(text))

    # The splits "a ", "b ", "c" (total length 5) are small enough to be
    # merged into a single chunk. This test ensures the final `if` block
    # in the merge function correctly captures these.
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "a b c"


def test_fixed_size_chunker_overlap_scan_completes():
    """
    Covers the `if start_of_overlap <= start_char:` line in FixedSizeChunker.
    This is hit when the overlap backward scan completes without breaking.
    """

    def length_function(text: str) -> int:
        return 10 if "X" in text else len(text)

    text = "abcXde"
    # chunk_size = 5, overlap = 4.
    # First chunk is "abc" because "X" is an oversized character.
    # The length of "abc" is 3, which is less than the overlap (4).
    # This allows the overlap scan to complete without breaking.
    chunker = FixedSizeChunker(
        chunk_size=5, chunk_overlap=4, length_function=length_function
    )
    chunks = list(chunker.chunk(text))

    # The sequence of chunks should be "abc", "X", "de".
    # This confirms that after chunking "abc", the next chunk starts at "X",
    # which happens when the full overlap scan triggers `start_char = end_char`.
    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "abc"
    assert chunks[1].text_for_generation == "X"
    assert chunks[2].text_for_generation == "de"


def test_fixed_size_chunker_handles_adjacent_oversized_char():
    """
    Tests correct chunking when a normal string is followed by an oversized char.
    The oversized char should be placed in its own chunk.
    """

    def length_function(text: str) -> int:
        return 100 if text == "b" else 1

    text = "ab"
    # chunk_size is 10.
    # The chunker should first create a chunk for "a".
    # Then, it will identify "b" as an oversized character and place it in
    # its own, separate chunk.
    chunker = FixedSizeChunker(chunk_size=10, length_function=length_function)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "a"
    assert chunks[1].text_for_generation == "b"
