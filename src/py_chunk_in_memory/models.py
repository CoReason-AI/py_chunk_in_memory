# Copyright (c) 2025 CoReason, Inc
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/py_chunk_in_memory

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


@dataclass
class Chunk:
    """
    Represents a standardized segment of text after chunking, enriched with
    metadata and relationship information as per the FRD.
    """

    text_for_generation: str
    chunk_id: UUID = field(default_factory=uuid4)

    # FRD R-5.1.3: Dual Text Fields
    text_for_embedding: Optional[str] = None

    # FRD R-5.2 & R-5.3: Standard Metadata and Relationships
    source_document_id: Optional[Any] = None
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

    # FRD R-2.1.5: Metadata Propagation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Element:
    """
    Represents a node in the Intermediate Document Representation (IDR),
    forming a tree structure that captures the document's hierarchy.
    """

    id: UUID = field(default_factory=uuid4)
    type: str = "text"
    text: str = ""
    parent: Optional["Element"] = field(default=None, repr=False)
    children: List["Element"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "Element"):
        """Adds a child element and sets its parent to this element."""
        child.parent = self
        self.children.append(child)
