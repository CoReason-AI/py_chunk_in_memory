# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from typing import Callable

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


def get_tiktoken_counter(
    model_name: str, special_tokens_handling: bool = False
) -> Callable[[str], int]:
    """
    Returns a function that counts tokens for a given text using a tiktoken model.

    Args:
        model_name: The name of the tiktoken model to use.
        special_tokens_handling: If True, accounts for special tokens in the count.

    Returns:
        A function that takes a string and returns the number of tokens.
    """
    if tiktoken is None:
        raise ImportError(
            "tiktoken is not installed. Please install it with `pip install py_chunk_in_memory[tokenizers]`"
        )

    encoding = tiktoken.get_encoding(model_name)

    def counter(text: str) -> int:
        if special_tokens_handling:
            return len(encoding.encode(text, allowed_special="all"))
        return len(encoding.encode(text, disallowed_special=()))

    return counter


def get_huggingface_counter(
    model_name_or_path: str, special_tokens_handling: bool = True
) -> Callable[[str], int]:
    """
    Returns a function that counts tokens for a given text using a Hugging Face model.

    Args:
        model_name_or_path: The name or path of the Hugging Face model to use.
        special_tokens_handling: If True (default), accounts for special tokens.

    Returns:
        A function that takes a string and returns the number of tokens.
    """
    if AutoTokenizer is None:
        raise ImportError(
            "transformers is not installed. Please install it with `pip install py_chunk_in_memory[tokenizers]`"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=special_tokens_handling))

    return counter
