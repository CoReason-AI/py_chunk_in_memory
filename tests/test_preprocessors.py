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
from py_chunk_in_memory.preprocessors import PreprocessorStep


def test_preprocessor_step_abc_cannot_be_instantiated():
    """Verify that the abstract PreprocessorStep class cannot be instantiated."""
    with pytest.raises(TypeError):
        PreprocessorStep()


def test_preprocessor_step_abc_process_raises_not_implemented():
    """Verify that the process method of a minimal subclass raises NotImplementedError."""

    class MinimalPreprocessor(PreprocessorStep):
        def process(self, text: str) -> str:
            # The type ignore is necessary because mypy flags this as an unsafe
            # super() call on an abstract method, which is the exact thing
            # we want to test.
            return super().process(text)  # type: ignore [safe-super]

    preprocessor = MinimalPreprocessor()
    with pytest.raises(NotImplementedError):
        preprocessor.process("test")
