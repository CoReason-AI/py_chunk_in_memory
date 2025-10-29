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
from py_chunk_in_memory.chunkers import BaseChunker


def test_base_chunker_cannot_be_instantiated():
    """Verify that the abstract BaseChunker cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseChunker"):
        BaseChunker()


def test_base_chunker_chunk_raises_not_implemented():
    """Verify calling a non-implemented chunk method raises NotImplementedError."""

    class ConcreteChunker(BaseChunker):
        """A concrete class for testing the abstract method invocation."""

        def chunk(self, text: str, **kwargs):
            return super().chunk(text, **kwargs)  # type: ignore[safe-super]

    chunker = ConcreteChunker()
    with pytest.raises(NotImplementedError):
        chunker.chunk("some text")
