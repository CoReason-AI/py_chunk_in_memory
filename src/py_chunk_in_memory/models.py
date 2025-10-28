# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from uuid import UUID, uuid4


@dataclass
class Chunk:
    """Represents a standardized segment of text after chunking."""

    text_for_generation: str
    text_for_embedding: Optional[str] = None
    chunk_id: UUID = field(default_factory=uuid4)
    source_document_id: UUID = field(default_factory=uuid4)
    previous_chunk_id: Optional[UUID] = None
    next_chunk_id: Optional[UUID] = None
    parent_chunk_id: Optional[UUID] = None
    start_char_index: int = 0
    end_char_index: int = 0
    sequence_number: int = 0
    token_count: int = 0
    content_type: str = "text"
    chunking_strategy_used: str = "unknown"
    hierarchical_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
