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

from py_chunk_in_memory.parsers import IDRParser


def test_idr_parser_cannot_be_instantiated() -> None:
    """Verify that the abstract BaseChunker class cannot be instantiated."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class IDRParser"):
        IDRParser()  # type: ignore[abstract]


def test_idr_parser_subclass_must_implement_parse() -> None:
    """Verify that a subclass of IDRParser must implement the parse method."""

    class IncompleteParser(IDRParser):
        pass

    with pytest.raises(
        TypeError, match="Can't instantiate abstract class IncompleteParser"
    ):
        IncompleteParser()  # type: ignore[abstract]


def test_idr_parser_parse_method_raises_not_implemented() -> None:
    """Verify that the abstract chunk method raises NotImplementedError."""

    class ConcreteParser(IDRParser):
        def parse(self, text: str) -> str:
            return super().parse(text)  # type: ignore [safe-super]

    parser = ConcreteParser()
    with pytest.raises(NotImplementedError):
        parser.parse("some text")
