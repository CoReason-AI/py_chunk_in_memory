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
from typing import Any, Dict, List, Optional

from py_chunk_in_memory.models import Element

try:
    import mistune
    from mistune.renderers.base import BaseRenderer
except ImportError:
    mistune = None
    BaseRenderer = object


class IDRParser(ABC):
    """Abstract base class for a parser that converts a document into an
    Intermediate Document Representation (IDR)."""

    @abstractmethod
    def parse(self, text: str) -> Element:
        """
        Parses the input text and returns an Intermediate Document Representation.

        Args:
            text: The input text to parse.

        Returns:
            The root Element of the parsed Intermediate Document Representation tree.
        """
        raise NotImplementedError


class _IDRRenderer(BaseRenderer):  # type: ignore[misc]
    """
    A Mistune renderer that constructs an Intermediate Document Representation (IDR)
    tree of Element objects from a Markdown document.
    """

    def __init__(self) -> None:
        super().__init__()
        self.root = Element(type="root")
        self.stack: List[Element] = [self.root]

    def __call__(
        self, tokens: List[Dict[str, Any]], state: Dict[str, Any]
    ) -> Element:
        self.render(tokens, state)
        return self.root

    def _get_current_parent(self) -> Element:
        return self.stack[-1]

    def render_children(self, token: Dict[str, Any], state: Dict[str, Any]) -> None:
        # The element for the current token is the last one added to its parent
        element = self._get_current_parent().children[-1]
        self.stack.append(element)
        super().render_children(token, state)
        self.stack.pop()

    def text(self, text: str, state: Dict[str, Any]) -> None:
        parent = self._get_current_parent()
        if parent.children and parent.children[-1].type == "text":
            parent.children[-1].text += text
        else:
            parent.add_child(Element(type="text", text=text))

    def paragraph(self, text: str, state: Dict[str, Any]) -> None:
        # This method's `text` argument contains the rendered children.
        # We create the paragraph element, and render_children will populate it.
        element = Element(type="paragraph")
        self._get_current_parent().add_child(element)

    def heading(self, text: str, level: int, state: Dict[str, Any]) -> None:
        element = Element(type=f"h{level}")
        self._get_current_parent().add_child(element)

    def list(self, text: str, ordered: bool, state: Dict[str, Any]) -> None:
        list_type = "ol" if ordered else "ul"
        element = Element(type=list_type)
        self._get_current_parent().add_child(element)

    def list_item(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="li")
        self._get_current_parent().add_child(element)

    def block_code(
        self, code: str, info: Optional[str], state: Dict[str, Any]
    ) -> None:
        element = Element(type="code_block", text=code)
        if info:
            element.metadata = {"language": info.strip()}
        self._get_current_parent().add_child(element)

    def block_quote(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="blockquote")
        self._get_current_parent().add_child(element)

    def table(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="table")
        self._get_current_parent().add_child(element)

    def table_row(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="table_row")
        self._get_current_parent().add_child(element)

    def table_cell(
        self, text: str, align: Optional[str], head: bool, state: Dict[str, Any]
    ) -> None:
        cell_type = "table_header" if head else "table_cell"
        element = Element(type=cell_type)
        if align:
            element.metadata = {"align": align}
        self._get_current_parent().add_child(element)

    def emphasis(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="emphasis", text=text)
        self._get_current_parent().add_child(element)

    def strong(self, text: str, state: Dict[str, Any]) -> None:
        element = Element(type="strong", text=text)
        self._get_current_parent().add_child(element)

    def link(
        self,
        link: str,
        text: Optional[str] = None,
        title: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        element = Element(type="link", text=text or "")
        element.metadata = {"url": link}
        if title:
            element.metadata["title"] = title
        self._get_current_parent().add_child(element)

    def image(
        self,
        src: str,
        alt: str = "",
        title: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        element = Element(type="image", text=alt)
        element.metadata = {"src": src}
        if title:
            element.metadata["title"] = title
        self._get_current_parent().add_child(element)

    def codespan(self, text: str, state: Optional[Dict[str, Any]] = None) -> None:
        self._get_current_parent().add_child(Element(type="codespan", text=text))


class MarkdownParser(IDRParser):
    """A parser for converting Markdown documents into an Intermediate Document
    Representation (IDR) tree."""

    def __init__(self) -> None:
        if mistune is None:
            raise ImportError(
                "mistune is not installed. Please install it with "
                "`pip install py_chunk_in_memory[structured]`"
            )
        self.parser = mistune.create_markdown(renderer=_IDRRenderer())

    def parse(self, text: str) -> Element:
        """
        Parses Markdown text into an IDR tree.
        """
        return self.parser(text)
