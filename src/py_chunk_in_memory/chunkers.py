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
from typing import Any, Iterable, List

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


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size segments."""

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, **kwargs: Any) -> Iterable[Chunk]:
        """
        Splits a text into fixed-size chunks.

        Args:
            text: The input text to be chunked.
            **kwargs: Additional parameters (not used in this implementation).

        Returns:
            An iterable of Chunk objects.
        """
        if not text:
            return []

        chunks: List[Chunk] = []
        start_index = 0
        sequence_number = 0

        while start_index < len(text):
            end_index = start_index + self.chunk_size
            chunk_text = text[start_index:end_index]

            chunk = Chunk(
                text_for_generation=chunk_text,
                start_char_index=start_index,
                end_char_index=start_index + len(chunk_text),
                sequence_number=sequence_number,
                chunking_strategy_used="fixed_size",
            )
            chunks.append(chunk)

            start_index += self.chunk_size - self.chunk_overlap
            sequence_number += 1

        return chunks
