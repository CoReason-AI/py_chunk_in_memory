# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

import pytest
from py_chunk_in_memory.chunkers import BaseChunker, FixedSizeChunker


def test_base_chunker_cannot_be_instantiated():
    """Verify that the abstract BaseChunker cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseChunker()


def test_base_chunker_chunk_raises_not_implemented():
    """Verify that calling chunk on a minimal subclass raises NotImplementedError."""

    class MinimalChunker(BaseChunker):
        def chunk(self, text: str, **kwargs):
            return super().chunk(text, **kwargs)

    chunker = MinimalChunker()
    with pytest.raises(NotImplementedError):
        chunker.chunk("test")


def test_fixed_size_chunker_basic():
    """Test basic fixed-size chunking with no overlap."""
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
    """Test fixed-size chunking with overlap."""
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
    assert chunks[2].start_char_index == 14 # 7 + (10 - 3)
    assert chunks[3].start_char_index == 21 # 14 + (10-3)

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
    """Verify that metadata fields are correctly populated."""
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
