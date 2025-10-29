# Copyright (c) 2025 Scientific Informatics, LLC
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory


from py_chunk_in_memory.models import Chunk


def test_chunk_text_for_generation_edge_cases():
    """Test edge cases for the `text_for_generation` field."""
    # Test with empty string
    chunk = Chunk(text_for_generation="")
    assert chunk.text_for_generation == ""

    # Test with very long string
    long_text = "a" * 10000
    chunk = Chunk(text_for_generation=long_text)
    assert chunk.text_for_generation == long_text


def test_chunk_text_for_embedding_edge_cases():
    """Test edge cases for the `text_for_embedding` field."""
    # Test with None (default)
    chunk = Chunk(text_for_generation="test")
    assert chunk.text_for_embedding is None

    # Test with empty string
    chunk = Chunk(text_for_generation="test", text_for_embedding="")
    assert chunk.text_for_embedding == ""

    # Test with long string
    long_text = "b" * 10000
    chunk = Chunk(text_for_generation="test", text_for_embedding=long_text)
    assert chunk.text_for_embedding == long_text


def test_chunk_id_uniqueness():
    """Verify that chunk_id is unique for each new instance."""
    chunk1 = Chunk(text_for_generation="test1")
    chunk2 = Chunk(text_for_generation="test2")
    assert chunk1.chunk_id != chunk2.chunk_id


def test_chunk_source_document_id_uniqueness():
    """Verify that source_document_id is unique for each new instance."""
    chunk1 = Chunk(text_for_generation="test1")
    chunk2 = Chunk(text_for_generation="test2")
    assert chunk1.source_document_id != chunk2.source_document_id


def test_chunk_id_fields_optional():
    """Test that UUID fields are optional and default to None where applicable."""
    chunk = Chunk(text_for_generation="test")
    assert chunk.previous_chunk_id is None
    assert chunk.next_chunk_id is None
    assert chunk.parent_chunk_id is None


def test_chunk_char_indices_edge_cases():
    """Test edge cases for start and end character indices."""
    # Test with zero values (default)
    chunk = Chunk(text_for_generation="test")
    assert chunk.start_char_index == 0
    assert chunk.end_char_index == 0

    # Test with large numbers
    chunk = Chunk(
        text_for_generation="test", start_char_index=100000, end_char_index=200000
    )
    assert chunk.start_char_index == 100000
    assert chunk.end_char_index == 200000


def test_chunk_sequence_number_edge_cases():
    """Test edge cases for sequence_number."""
    # Test with zero (default)
    chunk = Chunk(text_for_generation="test")
    assert chunk.sequence_number == 0

    # Test with a large number
    chunk = Chunk(text_for_generation="test", sequence_number=1000)
    assert chunk.sequence_number == 1000


def test_chunk_token_count_edge_cases():
    """Test edge cases for token_count."""
    # Test with zero (default)
    chunk = Chunk(text_for_generation="test")
    assert chunk.token_count == 0

    # Test with a large number
    chunk = Chunk(text_for_generation="test", token_count=5000)
    assert chunk.token_count == 5000


def test_chunk_content_type_edge_cases():
    """Test edge cases for content_type."""
    # Test with default "text"
    chunk = Chunk(text_for_generation="test")
    assert chunk.content_type == "text"

    # Test with different content types
    chunk = Chunk(text_for_generation="test", content_type="code")
    assert chunk.content_type == "code"


def test_chunk_chunking_strategy_used_edge_cases():
    """Test edge cases for chunking_strategy_used."""
    # Test with default "unknown"
    chunk = Chunk(text_for_generation="test")
    assert chunk.chunking_strategy_used == "unknown"

    # Test with a specific strategy
    chunk = Chunk(text_for_generation="test", chunking_strategy_used="semantic")
    assert chunk.chunking_strategy_used == "semantic"


def test_chunk_hierarchical_context_edge_cases():
    """Test edge cases for hierarchical_context."""
    # Test with default empty dict
    chunk = Chunk(text_for_generation="test")
    assert chunk.hierarchical_context == {}

    # Test with nested dictionary
    context = {"level1": {"level2": "value"}}
    chunk = Chunk(text_for_generation="test", hierarchical_context=context)
    assert chunk.hierarchical_context == context


def test_chunk_metadata_edge_cases():
    """Test edge cases for metadata."""
    # Test with default empty dict
    chunk = Chunk(text_for_generation="test")
    assert chunk.metadata == {}

    # Test with complex metadata
    metadata = {"source": "doc.txt", "tags": ["a", "b"], "version": 1.2}
    chunk = Chunk(text_for_generation="test", metadata=metadata)
    assert chunk.metadata == metadata
