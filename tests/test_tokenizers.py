# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

import sys
from unittest.mock import patch

import pytest

# Pre-import to ensure we can re-import it later
import py_chunk_in_memory.tokenizers as tokenizers_module

# Mark all tests in this module as requiring the 'tokenizers' extra
pytestmark = pytest.mark.tokenizers


@pytest.fixture(autouse=True)
def hide_optional_imports(monkeypatch):
    """Fixture to simulate optional dependencies not being installed."""
    # This fixture will be used to test the ImportError branches
    pass


def test_get_tiktoken_counter_success():
    """Verify the tiktoken counter works with a known model and text."""
    from py_chunk_in_memory.tokenizers import get_tiktoken_counter

    counter = get_tiktoken_counter("cl100k_base")
    text = "Hello, world! This is a test."
    # Expected token count for cl100k_base for the given text
    expected_tokens = 9
    assert counter(text) == expected_tokens


def test_get_huggingface_counter_success():
    """Verify the Hugging Face counter works with a known model and text."""
    from py_chunk_in_memory.tokenizers import get_huggingface_counter

    counter = get_huggingface_counter("gpt2")
    text = "Hello, world! This is a test."
    # Expected token count for gpt2 for the given text
    expected_tokens = 9
    assert counter(text) == expected_tokens


def test_get_tiktoken_counter_empty_string():
    """Verify tiktoken counter handles empty strings."""
    from py_chunk_in_memory.tokenizers import get_tiktoken_counter

    counter = get_tiktoken_counter("cl100k_base")
    assert counter("") == 0


def test_get_huggingface_counter_empty_string():
    """Verify Hugging Face counter handles empty strings."""
    from py_chunk_in_memory.tokenizers import get_huggingface_counter

    counter = get_huggingface_counter("gpt2")
    assert counter("") == 0


def test_tiktoken_import_error():
    """Verify an ImportError is raised if tiktoken is not installed."""
    with patch.dict(sys.modules, {"tiktoken": None}):
        # Need to reload the module to trigger the import error check
        from importlib import reload

        reload(tokenizers_module)

        with pytest.raises(ImportError, match="tiktoken is not installed"):
            tokenizers_module.get_tiktoken_counter("cl100k_base")
    # Reload again to restore the original state for other tests
    from importlib import reload

    reload(tokenizers_module)


def test_huggingface_import_error():
    """Verify an ImportError is raised if transformers is not installed."""
    with patch.dict(sys.modules, {"transformers": None}):
        # Need to reload the module to trigger the import error check
        from importlib import reload

        reload(tokenizers_module)

        with pytest.raises(ImportError, match="transformers is not installed"):
            tokenizers_module.get_huggingface_counter("gpt2")
    # Reload again to restore the original state for other tests
    from importlib import reload

    reload(tokenizers_module)
