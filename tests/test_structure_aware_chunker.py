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
from py_chunk_in_memory.chunkers import StructureAwareChunker
from py_chunk_in_memory.parsers import IDRParser
from py_chunk_in_memory.models import Element


class MockParser(IDRParser):
    """A mock parser for testing purposes."""
    def parse(self, text: str) -> Element:
        return Element(type="root", text=text)


def test_structure_aware_chunker_raises_error_without_parser():
    """Verify that instantiating StructureAwareChunker without a parser raises a TypeError."""
    with pytest.raises(TypeError):
        StructureAwareChunker()  # type: ignore


def test_structure_aware_chunker_raises_error_with_invalid_parser():
    """Verify that instantiating with a non-IDRParser object raises a TypeError."""
    with pytest.raises(TypeError, match="parser must be an instance of IDRParser."):
        StructureAwareChunker(parser=object(), chunk_size=100)


def test_structure_aware_chunker_chunk_raises_not_implemented():
    """Verify that calling the .chunk() method raises NotImplementedError."""
    parser = MockParser()
    chunker = StructureAwareChunker(parser=parser, chunk_size=100)
    with pytest.raises(NotImplementedError):
        chunker.chunk("some text")

def test_structure_aware_chunker_invalid_constructor_args():
    """Test that the StructureAwareChunker constructor validates its arguments."""
    parser = MockParser()
    with pytest.raises(ValueError, match="chunk_size must be a positive integer."):
        StructureAwareChunker(parser=parser, chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer."):
        StructureAwareChunker(parser=parser, chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size."):
        StructureAwareChunker(parser=parser, chunk_size=10, chunk_overlap=10)
