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
from typing import Any


class IDRParser(ABC):
    """
    Abstract base class for a parser that converts a text document into an
    Intermediate Document Representation (IDR).
    """

    @abstractmethod
    def parse(self, text: str) -> Any:
        """
        Parses the input text and returns an Intermediate Document Representation.

        Args:
            text: The input text to parse.

        Returns:
            An Intermediate Document Representation of the parsed text.
        """
        raise NotImplementedError
