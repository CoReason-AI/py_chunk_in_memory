# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 3_day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from abc import ABC, abstractmethod
from typing import Any, Iterable

from py_chunk_in_memory.models import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, **kwargs: Any) -> Iterable[Chunk]:
        """
        Splits a text into a sequence of chunks.

        Args:
            text: The input text to be chunked.
            **kwargs: Additional parameters specific to the chunking strategy.

        Returns:
            An iterable of Chunk objects.
        """
        raise NotImplementedError
