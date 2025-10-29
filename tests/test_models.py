# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

import uuid
from uuid import UUID
from py_chunk_in_memory.models import Chunk


def test_chunk_instantiation_defaults():
    """Verify a Chunk can be created with minimal arguments and defaults are correct."""
    text = "This is a test chunk."
    chunk = Chunk(text_for_generation=text)

    assert chunk.text_for_generation == text
    assert isinstance(chunk.chunk_id, UUID)
    assert isinstance(chunk.source_document_id, UUID)
    assert chunk.start_char_index == 0
    assert chunk.end_char_index == 0
    assert chunk.sequence_number == 0
    assert chunk.token_count == 0
    assert chunk.content_type == "text"
    assert chunk.chunking_strategy_used == "unknown"
    assert chunk.metadata == {}


def test_chunk_instantiation_with_all_fields():
    """Verify a Chunk can be created with all fields specified."""
    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    prev_id = uuid.uuid4()
    next_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    metadata = {"source": "test.txt"}
    h_context = {"H1": "Title"}

    chunk = Chunk(
        text_for_generation="Another test chunk.",
        text_for_embedding="embed text",
        chunk_id=chunk_id,
        source_document_id=doc_id,
        previous_chunk_id=prev_id,
        next_chunk_id=next_id,
        parent_chunk_id=parent_id,
        start_char_index=10,
        end_char_index=30,
        sequence_number=1,
        token_count=5,
        content_type="table",
        chunking_strategy_used="fixed_size",
        hierarchical_context=h_context,
        metadata=metadata,
    )

    assert chunk.text_for_generation == "Another test chunk."
    assert chunk.text_for_embedding == "embed text"
    assert chunk.chunk_id == chunk_id
    assert chunk.source_document_id == doc_id
    assert chunk.previous_chunk_id == prev_id
    assert chunk.next_chunk_id == next_id
    assert chunk.parent_chunk_id == parent_id
    assert chunk.start_char_index == 10
    assert chunk.end_char_index == 30
    assert chunk.sequence_number == 1
    assert chunk.token_count == 5
    assert chunk.content_type == "table"
    assert chunk.chunking_strategy_used == "fixed_size"
    assert chunk.hierarchical_context == h_context
    assert chunk.metadata == metadata


def test_chunk_mutable_defaults_are_independent():
    """Verify that mutable default fields are not shared between instances."""
    chunk1 = Chunk(text_for_generation="test1")
    chunk2 = Chunk(text_for_generation="test2")

    # Modify metadata of the first chunk
    chunk1.metadata["source"] = "doc1.txt"

    # Ensure the second chunk's metadata is not affected
    assert "source" not in chunk2.metadata
    assert chunk1.metadata != chunk2.metadata

    # Modify hierarchical_context of the first chunk
    chunk1.hierarchical_context["H1"] = "Title 1"

    # Ensure the second chunk's hierarchical_context is not affected
    assert "H1" not in chunk2.hierarchical_context
    assert chunk1.hierarchical_context != chunk2.hierarchical_context


def test_chunk_id_and_source_id_are_unique():
    """Verify that chunk_id and source_document_id are unique for new instances."""
    chunk1 = Chunk(text_for_generation="test1")
    chunk2 = Chunk(text_for_generation="test2")

    assert chunk1.chunk_id != chunk2.chunk_id
    assert chunk1.source_document_id != chunk2.source_document_id
