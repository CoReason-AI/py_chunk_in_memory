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
from py_chunk_in_memory.chunkers import (
    RecursiveCharacterChunker,
    FixedSizeChunker,
    SentenceChunker,
)


def test_recursive_chunker_empty_string_recursion_edge_case():
    """
    Test a subtle edge case in RecursiveCharacterChunker where a separator
    can create an empty string, and a custom length function can define that
    empty string's length as > chunk_size. This could cause infinite recursion
    if the chunker tries to split the oversized empty string again.
    """

    def custom_length_function(text: str) -> int:
        # Define the length of an empty string to be very large
        if text == "":
            return 1000
        return len(text)

    text = "a--b"
    # The separator "--" will split the text into ["a", "", "b"].
    # The middle empty string "" will have a length of 1000, exceeding chunk_size.
    # The chunker should handle this gracefully, not enter an infinite loop.
    chunker = RecursiveCharacterChunker(
        chunk_size=10,
        separators=["--"],
        keep_separator=False,
        length_function=custom_length_function,
    )
    chunks = list(chunker.chunk(text))

    # The expected behavior is that the oversized empty string is handled,
    # and the valid text parts are chunked correctly. The empty string itself
    # might be discarded or chunked, but the process must terminate.
    # Based on the logic, the non-empty parts "a" and "b" should form a chunk.
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "ab"


def test_fixed_size_chunker_overlap_greater_than_chunk_content():
    """
    Test a subtle edge case in FixedSizeChunker where a chunk's content
    is smaller than the configured chunk_overlap. The overlap logic should
    handle this by effectively starting the next chunk immediately after the
    previous one.
    """
    # A length function where 'X' is an oversized character
    def length_function(text: str) -> int:
        return 100 if "X" in text else len(text)

    text = "abcXde"
    # The first chunk will be "abc" (length 3), because "X" is oversized.
    # The chunk_overlap is 4, which is greater than the length of "abc".
    # The overlap scan should complete, and the next chunk should start
    # at the end of the previous chunk, which is at index 3 (the 'X').
    chunker = FixedSizeChunker(
        chunk_size=5, chunk_overlap=4, length_function=length_function
    )
    chunks = list(chunker.chunk(text))

    # The chunker should produce three chunks: "abc", "X", and "de".
    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "abc"
    assert chunks[1].text_for_generation == "X"
    assert chunks[2].text_for_generation == "de"


def test_recursive_character_chunker_with_overlapping_separators():
    """
    Test how the RecursiveCharacterChunker handles overlapping separators.
    The expected behavior is that it will split on the first separator it
    encounters in the text, and then continue searching for separators in the
    remainder of the string.
    """
    text = "abce"
    # With separators ["ab", "bc"], the chunker should first split on "ab".
    # This will result in two splits: "" and "ce".
    chunker = RecursiveCharacterChunker(
        chunk_size=10,
        separators=["ab", "bc"],
        keep_separator=True,
    )
    chunks = list(chunker.chunk(text))

    # The text should be split into "ab" and "ce", and then merged.
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == "abce"


def test_sentence_chunker_with_quoted_sentences():
    """
    Test that the SentenceChunker correctly handles sentences that contain
    quoted text with sentence terminators. The chunker should not split
    inside the quotes.
    """
    text = 'She said, "Is it morning yet? I can\'t wait for breakfast!" and then went back to sleep.'
    chunker = SentenceChunker(chunk_size=1000)
    chunks = list(chunker.chunk(text))

    # The entire text should be considered a single sentence.
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == text
