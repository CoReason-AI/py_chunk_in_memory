# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple

from py_chunk_in_memory.models import Chunk


@dataclass
class _Sentence:
    """Internal representation of a sentence with its character indices."""

    text: str
    start_index: int
    end_index: int


@dataclass
class _Split:
    """Internal representation of a text split with its character indices."""

    text: str
    start_index: int
    end_index: int


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


class RecursiveCharacterChunker(BaseChunker):
    """
    Recursively splits text based on a list of separators. This implementation
    is modeled after LangChain's popular text splitter, aiming for robustness
    and predictability. It now correctly tracks character indices.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = len,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        super().__init__(length_function=length_function)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        self.keep_separator = keep_separator

    def _split_text_with_indices(
        self, text: str, separators: List[str], initial_offset: int = 0
    ) -> List[_Split]:
        """
        Splits a text based on separators while keeping track of original indices.
        """
        final_splits: List[_Split] = []
        if not text:
            return final_splits

        separator = separators[0]
        remaining_separators = separators[1:]

        if not separator:  # Fallback for the last separator
            for i in range(0, len(text), self.chunk_size):
                segment = text[i : i + self.chunk_size]
                final_splits.append(
                    _Split(
                        text=segment,
                        start_index=initial_offset + i,
                        end_index=initial_offset + i + len(segment),
                    )
                )
            return final_splits

        try:
            last_end = 0
            for match in re.finditer(re.escape(separator), text):
                start, end = match.span()
                if self.keep_separator:
                    final_splits.append(
                        _Split(
                            text=text[last_end:end],
                            start_index=initial_offset + last_end,
                            end_index=initial_offset + end,
                        )
                    )
                else:
                    final_splits.append(
                        _Split(
                            text=text[last_end:start],
                            start_index=initial_offset + last_end,
                            end_index=initial_offset + start,
                        )
                    )
                last_end = end
            if last_end < len(text):
                final_splits.append(
                    _Split(
                        text=text[last_end:],
                        start_index=initial_offset + last_end,
                        end_index=initial_offset + len(text),
                    )
                )
        except re.error:  # Fallback for invalid regex
            for char_idx, char in enumerate(text):
                final_splits.append(
                    _Split(
                        text=char,
                        start_index=initial_offset + char_idx,
                        end_index=initial_offset + char_idx + 1,
                    )
                )

        # Recursively split any oversized chunks
        recursive_splits: List[_Split] = []
        for split in final_splits:
            if self._length_function(split.text) <= self.chunk_size:
                recursive_splits.append(split)
            else:
                next_separators = remaining_separators or [""]
                recursive_splits.extend(
                    self._split_text_with_indices(
                        split.text, next_separators, split.start_index
                    )
                )
        return recursive_splits

    def _merge_splits_with_indices(
        self, splits: List[_Split]
    ) -> List[Tuple[str, int, int]]:
        """
        Merges a list of _Split objects into larger chunks.
        """
        final_chunks: List[Tuple[str, int, int]] = []
        current_chunk_parts: List[_Split] = []
        current_length = 0

        for split in splits:
            split_len = self._length_function(split.text)
            if current_length + split_len > self.chunk_size and current_chunk_parts:
                start_index = current_chunk_parts[0].start_index
                end_index = current_chunk_parts[-1].end_index
                final_chunks.append(
                    (
                        "".join(p.text for p in current_chunk_parts),
                        start_index,
                        end_index,
                    )
                )

                # Determine overlap
                overlap_parts: List[_Split] = []
                overlap_len = 0
                for i in range(len(current_chunk_parts) - 1, -1, -1):
                    part = current_chunk_parts[i]
                    part_len = self._length_function(part.text)
                    if overlap_len + part_len > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len

                current_chunk_parts = overlap_parts
                current_length = overlap_len

            current_chunk_parts.append(split)
            current_length += split_len

        if current_chunk_parts:
            start_index = current_chunk_parts[0].start_index
            end_index = current_chunk_parts[-1].end_index
            final_chunks.append(
                ("".join(p.text for p in current_chunk_parts), start_index, end_index)
            )

        return final_chunks

    def chunk(self, text: str, **kwargs: Any) -> Iterable[Chunk]:
        if not text:
            return []

        splits = self._split_text_with_indices(text, self.separators)
        chunk_data = self._merge_splits_with_indices(splits)

        final_chunks = []
        for i, (chunk_text, start_index, end_index) in enumerate(chunk_data):
            final_chunks.append(
                Chunk(
                    text_for_generation=chunk_text,
                    start_char_index=start_index,
                    end_char_index=end_index,
                    sequence_number=i,
                    token_count=self._length_function(chunk_text),
                    chunking_strategy_used="recursive_character",
                )
            )
        return final_chunks


class SentenceChunker(BaseChunker):
    """
    Splits text into sentences and aggregates them into chunks of a specified size.
    This chunker provides a robust regex-based fallback if NLP libraries are not
    available.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        length_function: Callable[[str], int] = len,
        overlap_sentences: int = 0,
    ):
        """
        Initializes the SentenceChunker.
        Args:
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The overlap between consecutive chunks (token-based).
            length_function: The function to measure text size.
            overlap_sentences: The number of sentences to overlap. If > 0, this
                               overrides `chunk_overlap`.
        """
        super().__init__(length_function=length_function)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be a non-negative integer.")
        if chunk_overlap >= chunk_size and overlap_sentences == 0:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be a non-negative integer.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.overlap_sentences = overlap_sentences

    def _split_text_with_regex(self, text: str) -> List[_Sentence]:
        """
        A robust regex-based sentence splitter that preserves character indices.
        """
        sentences = []
        pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

        last_end = 0
        for match in re.finditer(pattern, text):
            raw_sentence = text[last_end : match.start()]
            stripped_sentence = raw_sentence.strip()
            if stripped_sentence:
                start_index = last_end + raw_sentence.find(stripped_sentence)
                end_index = start_index + len(stripped_sentence)
                sentences.append(_Sentence(stripped_sentence, start_index, end_index))
            last_end = match.end()

        remaining_text = text[last_end:]
        stripped_remaining = remaining_text.strip()
        if stripped_remaining:
            start_index = last_end + remaining_text.find(stripped_remaining)
            end_index = start_index + len(stripped_remaining)
            sentences.append(_Sentence(stripped_remaining, start_index, end_index))

        return sentences

    def _merge_sentences(
        self, sentences: List[_Sentence]
    ) -> List[Tuple[str, int, int]]:
        """
        Merges a list of _Sentence objects into chunks.
        Returns a list of tuples, where each tuple contains the chunk text,
        start index, and end index.
        """
        final_chunks: List[Tuple[str, int, int]] = []
        current_chunk_sentences: List[_Sentence] = []

        def join_sentence_texts(sents: List[_Sentence]) -> str:
            return " ".join(s.text for s in sents)

        for sentence in sentences:
            if self._length_function(sentence.text) > self.chunk_size:
                if current_chunk_sentences:
                    start_idx = current_chunk_sentences[0].start_index
                    end_idx = current_chunk_sentences[-1].end_index
                    final_chunks.append(
                        (
                            join_sentence_texts(current_chunk_sentences),
                            start_idx,
                            end_idx,
                        )
                    )
                current_chunk_sentences = []
                final_chunks.append(
                    (sentence.text, sentence.start_index, sentence.end_index)
                )
                continue

            prospective_chunk = current_chunk_sentences + [sentence]
            if (
                self._length_function(join_sentence_texts(prospective_chunk))
                > self.chunk_size
            ):
                if current_chunk_sentences:
                    start_idx = current_chunk_sentences[0].start_index
                    end_idx = current_chunk_sentences[-1].end_index
                    final_chunks.append(
                        (
                            join_sentence_texts(current_chunk_sentences),
                            start_idx,
                            end_idx,
                        )
                    )

                if self.overlap_sentences > 0:
                    overlap_sents = current_chunk_sentences[-self.overlap_sentences :]
                else:
                    token_overlap_sents: List[_Sentence] = []
                    for i in range(len(current_chunk_sentences) - 1, -1, -1):
                        sent = current_chunk_sentences[i]
                        prospective_overlap = [sent] + token_overlap_sents
                        if (
                            self._length_function(
                                join_sentence_texts(prospective_overlap)
                            )
                            > self.chunk_overlap
                        ):
                            break
                        token_overlap_sents = prospective_overlap
                    overlap_sents = token_overlap_sents

                current_chunk_sentences = overlap_sents
                if (
                    self._length_function(
                        join_sentence_texts(current_chunk_sentences + [sentence])
                    )
                    <= self.chunk_size
                ):
                    current_chunk_sentences.append(sentence)
                else:
                    # If the new sentence can't fit even with the overlap, start a new chunk
                    final_chunks.append(
                        (sentence.text, sentence.start_index, sentence.end_index)
                    )
                    current_chunk_sentences = []  # Reset for next iteration
            else:
                current_chunk_sentences = prospective_chunk

        if current_chunk_sentences:
            start_idx = current_chunk_sentences[0].start_index
            end_idx = current_chunk_sentences[-1].end_index
            final_chunks.append(
                (join_sentence_texts(current_chunk_sentences), start_idx, end_idx)
            )

        return final_chunks

    def chunk(self, text: str, **kwargs: Any) -> Iterable[Chunk]:
        """
        Chunks the text by splitting it into sentences and then merging them.
        """
        if not text:
            return []

        sentences = self._split_text_with_regex(text)
        chunk_data = self._merge_sentences(sentences)

        final_chunks = []
        for i, (chunk_text, start_index, end_index) in enumerate(chunk_data):
            final_chunks.append(
                Chunk(
                    text_for_generation=chunk_text,
                    start_char_index=start_index,
                    end_char_index=end_index,
                    sequence_number=i,
                    token_count=self._length_function(chunk_text),
                    chunking_strategy_used="sentence",
                )
            )
        return final_chunks
