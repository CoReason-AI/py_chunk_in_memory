# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

"""Tests for the data models in the package."""

import uuid
from py_chunk_in_memory.models import Chunk, Element


def test_chunk_creation_defaults():
    """Verify that a Chunk can be created with minimal arguments and defaults are set."""
    text = "This is a test chunk."
    chunk = Chunk(text_for_generation=text)

    assert chunk.text_for_generation == text
    assert isinstance(chunk.chunk_id, uuid.UUID)
    assert chunk.text_for_embedding is None
    assert chunk.source_document_id is None
    assert chunk.previous_chunk_id is None
    assert chunk.next_chunk_id is None
    assert chunk.parent_chunk_id is None
    assert chunk.start_char_index == 0
    assert chunk.end_char_index == 0
    assert chunk.sequence_number == 0
    assert chunk.token_count == 0
    assert chunk.content_type == "text"
    assert chunk.chunking_strategy_used == "unknown"
    assert chunk.hierarchical_context == {}
    assert chunk.metadata == {}


def test_chunk_creation_with_all_fields():
    """Verify that a Chunk can be created with all fields specified."""
    chunk_id = uuid.uuid4()
    source_id = "doc-123"
    prev_id = uuid.uuid4()
    next_id = uuid.uuid4()
    parent_id = uuid.uuid4()

    chunk = Chunk(
        text_for_generation="Full text.",
        chunk_id=chunk_id,
        text_for_embedding="Embedding text.",
        source_document_id=source_id,
        previous_chunk_id=prev_id,
        next_chunk_id=next_id,
        parent_chunk_id=parent_id,
        start_char_index=10,
        end_char_index=20,
        sequence_number=5,
        token_count=4,
        content_type="table",
        chunking_strategy_used="semantic",
        hierarchical_context={"H1": "Title"},
        metadata={"custom": "value"},
    )

    assert chunk.chunk_id == chunk_id
    assert chunk.source_document_id == source_id
    assert chunk.previous_chunk_id == prev_id
    assert chunk.next_chunk_id == next_id
    assert chunk.parent_chunk_id == parent_id
    assert chunk.start_char_index == 10
    assert chunk.end_char_index == 20
    assert chunk.sequence_number == 5
    assert chunk.token_count == 4
    assert chunk.content_type == "table"
    assert chunk.chunking_strategy_used == "semantic"
    assert chunk.hierarchical_context == {"H1": "Title"}
    assert chunk.metadata == {"custom": "value"}


def test_chunk_default_factories_are_independent():
    """Verify that default factories create unique objects for each instance."""
    chunk1 = Chunk(text_for_generation="First chunk")
    chunk2 = Chunk(text_for_generation="Second chunk")

    assert chunk1.chunk_id != chunk2.chunk_id
    assert chunk1.metadata is not chunk2.metadata
    assert chunk1.hierarchical_context is not chunk2.hierarchical_context

    # Modify metadata of one to ensure it doesn't affect the other
    chunk1.metadata["key"] = "value"
    assert "key" not in chunk2.metadata


def test_chunk_edge_cases():
    """Test edge cases for the Chunk data model."""
    # Test with empty text
    empty_chunk = Chunk(text_for_generation="")
    assert empty_chunk.text_for_generation == ""

    # Test with various types for source_document_id
    chunk_with_int_id = Chunk(text_for_generation="text", source_document_id=123)
    assert chunk_with_int_id.source_document_id == 123

    chunk_with_str_id = Chunk(text_for_generation="text", source_document_id="doc-abc")
    assert chunk_with_str_id.source_document_id == "doc-abc"

    # Test with complex metadata
    complex_meta = {"source": "web", "nested": {"key": [1, 2, 3]}}
    chunk_with_meta = Chunk(text_for_generation="text", metadata=complex_meta)
    assert chunk_with_meta.metadata == complex_meta

    # Test with complex hierarchical context
    complex_context = {"H1": "Main Title", "H2": "Subtitle", "P": 1}
    chunk_with_context = Chunk(
        text_for_generation="text", hierarchical_context=complex_context
    )
    assert chunk_with_context.hierarchical_context == complex_context


def test_element_creation_defaults():
    """Verify that an Element can be created with minimal arguments and defaults are set."""
    element = Element()

    assert isinstance(element.id, uuid.UUID)
    assert element.type == "text"
    assert element.text == ""
    assert element.parent is None
    assert element.children == []


def test_element_creation_with_all_fields():
    """Verify that an Element can be created with all fields specified."""
    element_id = uuid.uuid4()
    parent = Element(type="root")
    child = Element(type="child")

    element = Element(
        id=element_id,
        type="paragraph",
        text="This is a paragraph.",
        parent=parent,
        children=[child],
    )

    assert element.id == element_id
    assert element.type == "paragraph"
    assert element.text == "This is a paragraph."
    assert element.parent == parent
    assert element.children == [child]


def test_element_add_child():
    """Verify the add_child method correctly establishes the parent-child relationship."""
    parent = Element(type="section")
    child1 = Element(type="p", text="First paragraph.")
    child2 = Element(type="p", text="Second paragraph.")

    parent.add_child(child1)
    parent.add_child(child2)

    assert len(parent.children) == 2
    assert parent.children[0] == child1
    assert parent.children[1] == child2
    assert child1.parent == parent
    assert child2.parent == parent


def test_element_default_factories_are_independent():
    """Verify that default factories create unique objects for each instance."""
    el1 = Element()
    el2 = Element()

    assert el1.id != el2.id
    assert el1.children is not el2.children

    # Modify children of one to ensure it doesn't affect the other
    el1.add_child(Element())
    assert len(el1.children) == 1
    assert len(el2.children) == 0
