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
from unittest.mock import patch

from py_chunk_in_memory.tokenizers import (
    get_huggingface_counter,
    get_tiktoken_counter,
)

# A common model used in many RAG applications
TIKTOKEN_MODEL_NAME = "cl100k_base"
# A common open-source model
HF_MODEL_NAME = "bert-base-uncased"


@pytest.mark.tokenizers
def test_get_tiktoken_counter_returns_callable():
    """Verify that get_tiktoken_counter returns a callable function."""
    counter = get_tiktoken_counter(TIKTOKEN_MODEL_NAME)
    assert callable(counter)


@pytest.mark.tokenizers
def test_get_tiktoken_counter_default_no_special_tokens():
    """Test the default behavior (no special tokens) for tiktoken."""
    counter = get_tiktoken_counter(TIKTOKEN_MODEL_NAME)
    text_with_special_token = "Hello<|endoftext|>"
    # Should count "Hello" and "<|endoftext|>" as 6 separate tokens: 'Hello', '<', '|', 'end', 'of', 'text', '|', '>'
    # The exact count can vary, but it should not be 2. Let's find the exact count without special tokens
    # encoding.encode("Hello<|endoftext|>", allowed_special=set()) -> [9906, 27, 91, 437, 27, 29] (6 tokens)
    assert counter(text_with_special_token) == 8


@pytest.mark.tokenizers
def test_get_tiktoken_counter_with_special_tokens():
    """Test tiktoken counting with special_tokens_handling enabled."""
    counter = get_tiktoken_counter(TIKTOKEN_MODEL_NAME, special_tokens_handling=True)
    text_with_special_token = "Hello<|endoftext|>"
    # Should count "Hello" as one token and "<|endoftext|>" as one special token.
    # encoding.encode("Hello<|endoftext|>", allowed_special="all") -> [9906, 100257] (2 tokens)
    assert counter(text_with_special_token) == 2


def test_get_tiktoken_import_error():
    """Verify ImportError is raised if tiktoken is not installed."""
    with patch("py_chunk_in_memory.tokenizers.tiktoken", None):
        with pytest.raises(ImportError, match="tiktoken is not installed"):
            get_tiktoken_counter(TIKTOKEN_MODEL_NAME)


@pytest.mark.tokenizers
def test_get_huggingface_counter_returns_callable():
    """Verify that get_huggingface_counter returns a callable function."""
    counter = get_huggingface_counter(HF_MODEL_NAME)
    assert callable(counter)


@pytest.mark.tokenizers
def test_get_huggingface_counter_default_with_special_tokens():
    """Test the default behavior (with special tokens) for Hugging Face."""
    counter = get_huggingface_counter(HF_MODEL_NAME)
    text = "Hello world"
    # [CLS] Hello world [SEP] -> 101, 7592, 2088, 102
    assert counter(text) == 4


@pytest.mark.tokenizers
def test_get_huggingface_counter_without_special_tokens():
    """Test Hugging Face counting with special_tokens_handling disabled."""
    counter = get_huggingface_counter(HF_MODEL_NAME, special_tokens_handling=False)
    text = "Hello world"
    # Hello world -> 7592, 2088
    assert counter(text) == 2


def test_get_huggingface_import_error():
    """Verify ImportError is raised if transformers is not installed."""
    with patch("py_chunk_in_memory.tokenizers.AutoTokenizer", None):
        with pytest.raises(ImportError, match="transformers is not installed"):
            get_huggingface_counter(HF_MODEL_NAME)
