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
from py_chunk_in_memory.chunkers import SentenceChunker


def test_sentence_chunker_invalid_params():
    """Test that the chunker raises errors for invalid parameters."""
    with pytest.raises(ValueError, match="chunk_size must be a positive integer."):
        SentenceChunker(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer."):
        SentenceChunker(chunk_size=10, chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size."):
        SentenceChunker(chunk_size=10, chunk_overlap=10)
    with pytest.raises(ValueError, match="overlap_sentences must be a non-negative integer."):
        SentenceChunker(chunk_size=10, overlap_sentences=-1)


def test_sentence_chunker_empty_string():
    """Test chunking an empty string returns no chunks."""
    chunker = SentenceChunker(chunk_size=100)
    chunks = list(chunker.chunk(""))
    assert len(chunks) == 0


def test_sentence_chunker_small_string():
    """Test a string smaller than the chunk size results in one chunk."""
    text = "This is a single sentence."
    chunker = SentenceChunker(chunk_size=100)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == text


def test_sentence_chunker_basic_chunking_no_overlap():
    """Test basic sentence splitting and merging without overlap."""
    text = "First sentence. Second sentence. Third sentence."
    chunker = SentenceChunker(chunk_size=35, chunk_overlap=0)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "First sentence. Second sentence."
    assert chunks[1].text_for_generation == "Third sentence."


def test_sentence_chunker_with_token_overlap():
    """Test sentence chunking with token-based overlap."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunker = SentenceChunker(chunk_size=30, chunk_overlap=15)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "Sentence one. Sentence two."
    assert chunks[1].text_for_generation == "Sentence two. Sentence three."
    assert chunks[2].text_for_generation == "Sentence three. Sentence four."


def test_sentence_chunker_with_sentence_overlap():
    """Test sentence chunking with sentence-based overlap."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunker = SentenceChunker(chunk_size=45, overlap_sentences=1)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "Sentence one. Sentence two. Sentence three."
    assert chunks[1].text_for_generation == "Sentence three. Sentence four. Sentence five."


def test_sentence_overlap_overrides_token_overlap():
    """Verify that overlap_sentences takes precedence over chunk_overlap."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    # chunk_overlap is ignored
    chunker = SentenceChunker(chunk_size=45, chunk_overlap=100, overlap_sentences=1)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "Sentence one. Sentence two. Sentence three."
    assert chunks[1].text_for_generation == "Sentence three. Sentence four. Sentence five."


def test_sentence_chunker_handles_abbreviations():
    """Verify the regex splitter does not split on common abbreviations."""
    text = "Dr. Smith went to Washington D.C. to see the U.S.A."
    chunker = SentenceChunker(chunk_size=100)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 1
    assert chunks[0].text_for_generation == text


def test_sentence_chunker_handles_various_terminators():
    """Test splitting with question marks and exclamation points."""
    text = "What is your name? My name is Jules! It is nice to meet you."
    chunker = SentenceChunker(chunk_size=40)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "What is your name? My name is Jules!"
    assert chunks[1].text_for_generation == "It is nice to meet you."


def test_sentence_chunker_long_sentence_fallback():
    """Test that a sentence longer than chunk_size becomes its own chunk."""
    long_sentence = "This is a very long sentence that will definitely exceed the chunk size."
    text = f"A short sentence. {long_sentence} Another short sentence."
    chunker = SentenceChunker(chunk_size=50)
    chunks = list(chunker.chunk(text))
    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "A short sentence."
    assert chunks[1].text_for_generation == long_sentence
    assert chunks[2].text_for_generation == "Another short sentence."


def test_sentence_chunker_metadata_population():
    """Verify that chunk metadata is populated correctly."""
    text = "Sentence one. Sentence two."
    chunker = SentenceChunker(chunk_size=20)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].sequence_number == 0
    assert chunks[0].chunking_strategy_used == "sentence"
    assert chunks[0].token_count == len(chunks[0].text_for_generation)
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 13

    assert chunks[1].sequence_number == 1
    assert chunks[1].chunking_strategy_used == "sentence"
    assert chunks[1].token_count == len(chunks[1].text_for_generation)
    assert chunks[1].start_char_index == 14
    assert chunks[1].end_char_index == 27


def test_sentence_chunker_with_custom_length_function_and_sentence_overlap():
    """Test the sentence chunker with a custom length function and sentence overlap."""

    def word_counter(text: str) -> int:
        return len(text.split())

    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    # Chunk into segments of <= 7 words, with an overlap of 1 sentence.
    chunker = SentenceChunker(chunk_size=7, overlap_sentences=1, length_function=word_counter)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 3
    assert chunks[0].text_for_generation == "One two three. Four five six."
    assert chunks[1].text_for_generation == "Four five six. Seven eight nine."
    assert chunks[2].text_for_generation == "Seven eight nine. Ten eleven twelve."


def test_no_data_loss_on_tricky_overlap():
    """
    Regression test for a data loss bug.
    This test ensures that a sentence is not dropped when the overlap from the
    previous chunk plus the new sentence exceeds the chunk_size.
    """
    text = "aaa. bbb. ccc. ddd eee fff."
    # Sentences lengths: 3, 3, 3, 11
    # 1. Chunk 1: "aaa. bbb. ccc." (11 chars)
    # 2. Overlap: "ccc." (3 chars)
    # 3. Next chunk starts with "ccc."
    # 4. Add "ddd eee fff." (11 chars). Total: 3 + 1 + 11 = 15. This is > chunk_size.
    #    The bug was that "ddd eee fff." would be dropped.
    #    The correct behavior is to start a new chunk with just "ddd eee fff."
    chunker = SentenceChunker(chunk_size=14, chunk_overlap=4)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    assert chunks[0].text_for_generation == "aaa. bbb. ccc."
    assert chunks[1].text_for_generation == "ddd eee fff."
    # Verify no data is lost
    full_text = " ".join([c.text_for_generation for c in chunks])
    assert "ddd eee fff." in full_text


def test_char_indices_with_varied_whitespace():
    """
    Test that start_char_index and end_char_index are correct even with
    varied whitespace in the original text.
    """
    text = "Sentence one.\n\nSentence two. \t Sentence three."
    chunker = SentenceChunker(chunk_size=30)
    chunks = list(chunker.chunk(text))

    assert len(chunks) == 2
    # Chunk 1: "Sentence one. Sentence two."
    # The text_for_generation has normalized whitespace.
    assert chunks[0].text_for_generation == "Sentence one. Sentence two."
    # The indices should reflect the original text.
    assert chunks[0].start_char_index == 0
    assert chunks[0].end_char_index == 28

    # Chunk 2: "Sentence three."
    assert chunks[1].text_for_generation == "Sentence three."
    assert chunks[1].start_char_index == 31
    assert chunks[1].end_char_index == 46
