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


def test_idr_parser_cannot_be_instantiated():
    """Verify that the IDRParser Abstract Base Class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        IDRParser()


def test_idr_parser_parse_raises_not_implemented():
    """Verify that calling parse on a minimal subclass raises NotImplementedError."""

    class MinimalParser(IDRParser):
        def parse(self, text: str):
            return super().parse(text)

    parser = MinimalParser()
    with pytest.raises(NotImplementedError):
        parser.parse("test")
