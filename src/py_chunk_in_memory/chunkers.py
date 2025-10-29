# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, List

from py_chunk_in_memory.models import Chunk


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    def __init__(self, length_function: Callable[[str], int] = len):
        """
        Initializes the BaseChunker.

        Args:
            length_function: A function that measures the size of a text string.
                             Defaults to `len`.
        """
        self._length_function = length_function

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
    """
    Chunks text into fixed-size segments based on a specified length function.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = len,
    ):
        """
        Initializes the FixedSizeChunker.

        Args:
            chunk_size: The maximum size of each chunk, measured by `length_function`.
            chunk_overlap: The desired overlap between consecutive chunks,
                           measured by `length_function`.
            length_function: The function to use for measuring text size.
        """
        super().__init__(length_function=length_function)
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
        Splits a text into fixed-size chunks respecting the `length_function`.
        """
        if not text:
            return []

        chunks: List[Chunk] = []
        start_char = 0
        sequence_number = 0
        text_len_chars = len(text)

        while start_char < text_len_chars:
            # 1. Find the next hard boundary (a character that is oversized by itself)
            next_oversized_char_index = -1
            for i in range(start_char, text_len_chars):
                if self._length_function(text[i : i + 1]) > self.chunk_size:
                    next_oversized_char_index = i
                    break

            # 2. Determine the scannable limit for the current chunk
            limit = (
                next_oversized_char_index
                if next_oversized_char_index != -1
                else text_len_chars
            )

            # 3. Find the end of the chunk
            end_char = start_char

            # If the oversized character is right at the start, it's our whole chunk
            if limit == start_char:
                end_char = start_char + 1
            else:
                # Scan forward to the limit to find the largest chunk that fits
                while end_char < limit:
                    if (
                        self._length_function(text[start_char : end_char + 1])
                        > self.chunk_size
                    ):
                        break
                    end_char += 1

            # 4. Create the chunk
            chunk_text = text[start_char:end_char]
            if not chunk_text:
                break  # Avoid creating empty chunks

            chunk = Chunk(
                text_for_generation=chunk_text,
                start_char_index=start_char,
                end_char_index=end_char,
                sequence_number=sequence_number,
                token_count=self._length_function(chunk_text),
                chunking_strategy_used="fixed_size",
            )
            chunks.append(chunk)
            sequence_number += 1

            # 5. Determine the start of the next chunk
            if end_char == text_len_chars:
                break

            # If the chunk ended right before an oversized char, start the next one there
            if next_oversized_char_index == end_char:
                start_char = end_char
            else:
                # Otherwise, calculate the start based on overlap
                start_of_overlap = end_char
                while start_of_overlap > start_char:
                    overlap_text = text[start_of_overlap - 1 : end_char]
                    if self._length_function(overlap_text) > self.chunk_overlap:
                        break
                    start_of_overlap -= 1

                if start_of_overlap <= start_char:
                    start_char = end_char
                else:
                    start_char = start_of_overlap
        return chunks
