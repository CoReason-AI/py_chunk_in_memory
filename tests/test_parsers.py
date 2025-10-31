# Copyright (c) 2025 CoReason, Inc
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

from py_chunk_in_memory.models import Element
from py_chunk_in_memory.parsers import IDRParser

# Try to import MarkdownParser; skip tests if it fails (e.g., mistune not installed)
try:
    from py_chunk_in_memory.parsers import MarkdownParser

    MISTUNE_INSTALLED = True
except ImportError:
    MISTUNE_INSTALLED = False


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
        def parse(self, text: str) -> Element:
            return super().parse(text)  # type: ignore[safe-super]

    parser = ConcreteParser()
    with pytest.raises(NotImplementedError):
        parser.parse("some text")


@pytest.mark.skipif(not MISTUNE_INSTALLED, reason="mistune is not installed")
class TestMarkdownParser:
    """Unit tests for the MarkdownParser."""

    def test_markdown_parser_initialization_raises_import_error(self) -> None:
        """
        Verify that MarkdownParser raises ImportError if mistune is not available.
        """
        with patch.dict(sys.modules, {"mistune": None}):
            # Need to reload the module to trigger the import error
            import importlib
            from py_chunk_in_memory import parsers

            importlib.reload(parsers)
            with pytest.raises(
                ImportError,
                match="mistune is not installed. Please install it with "
                "`pip install py_chunk_in_memory\\[structured\\]`",
            ):
                parsers.MarkdownParser()
            # Restore the original module
            importlib.reload(parsers)

    def test_simple_paragraph(self) -> None:
        """Test parsing of a simple paragraph."""
        parser = MarkdownParser()
        markdown_text = "This is a simple paragraph."
        root = parser.parse(markdown_text)

        assert root.type == "root"
        assert len(root.children) == 1
        paragraph = root.children[0]
        assert paragraph.type == "paragraph"
        assert paragraph.children[0].type == "text"
        assert paragraph.children[0].text == "This is a simple paragraph."

    def test_headings(self) -> None:
        """Test parsing of different heading levels."""
        parser = MarkdownParser()
        markdown_text = "# H1\n\n## H2"
        root = parser.parse(markdown_text)

        assert len(root.children) == 2
        h1 = root.children[0]
        assert h1.type == "h1"
        assert h1.children[0].text == "H1"

        h2 = root.children[1]
        assert h2.type == "h2"
        assert h2.children[0].text == "H2"

    def test_unordered_list(self) -> None:
        """Test parsing of an unordered list."""
        parser = MarkdownParser()
        markdown_text = "* Item 1\n* Item 2"
        root = parser.parse(markdown_text)

        assert len(root.children) == 1
        ul = root.children[0]
        assert ul.type == "ul"
        assert len(ul.children) == 2
        li1 = ul.children[0]
        assert li1.type == "li"
        assert li1.children[0].children[0].text == "Item 1"
        li2 = ul.children[1]
        assert li2.type == "li"
        assert li2.children[0].children[0].text == "Item 2"

    def test_ordered_list(self) -> None:
        """Test parsing of an ordered list."""
        parser = MarkdownParser()
        markdown_text = "1. First\n2. Second"
        root = parser.parse(markdown_text)

        assert len(root.children) == 1
        ol = root.children[0]
        assert ol.type == "ol"
        assert len(ol.children) == 2
        li1 = ol.children[0]
        assert li1.type == "li"
        assert li1.children[0].children[0].text == "First"
        li2 = ol.children[1]
        assert li2.type == "li"
        assert li2.children[0].children[0].text == "Second"

    def test_code_block(self) -> None:
        """Test parsing of a fenced code block with a language."""
        parser = MarkdownParser()
        markdown_text = "```python\nprint('hello')\n```"
        root = parser.parse(markdown_text)

        assert len(root.children) == 1
        code_block = root.children[0]
        assert code_block.type == "code_block"
        assert code_block.text == "print('hello')\n"
        assert code_block.metadata == {"language": "python"}

    def test_blockquote(self) -> None:
        """Test parsing of a blockquote."""
        parser = MarkdownParser()
        markdown_text = "> This is a quote."
        root = parser.parse(markdown_text)

        assert len(root.children) == 1
        blockquote = root.children[0]
        assert blockquote.type == "blockquote"
        assert blockquote.children[0].children[0].text == "This is a quote."

    def test_table(self) -> None:
        """Test parsing of a simple table."""
        parser = MarkdownParser()
        markdown_text = "| Head 1 | Head 2 |\n|--------|--------|\n| Cell 1 | Cell 2 |"
        root = parser.parse(markdown_text)

        assert len(root.children) == 1
        table = root.children[0]
        assert table.type == "table"

        header_row = table.children[0]
        assert header_row.type == "table_row"
        assert len(header_row.children) == 2
        assert header_row.children[0].type == "table_header"
        assert header_row.children[0].children[0].text == "Head 1"
        assert header_row.children[1].type == "table_header"
        assert header_row.children[1].children[0].text == "Head 2"

        body_row = table.children[1]
        assert body_row.type == "table_row"
        assert len(body_row.children) == 2
        assert body_row.children[0].type == "table_cell"
        assert body_row.children[0].children[0].text == "Cell 1"
        assert body_row.children[1].type == "table_cell"
        assert body_row.children[1].children[0].text == "Cell 2"

    def test_inline_formatting(self) -> None:
        """Test parsing of various inline formatting elements."""
        parser = MarkdownParser()
        markdown_text = "This is **bold**, *italic*, and `code`."
        root = parser.parse(markdown_text)

        paragraph = root.children[0]
        assert paragraph.type == "paragraph"

        # This is a bit complex due to how mistune creates text and inline elements
        # A simpler approach is to check the concatenated text content

        text_parts = [
            child.text for child in paragraph.children if hasattr(child, "text")
        ]
        assert "".join(text_parts) == "This is bold, italic, and code."

        element_types = [child.type for child in paragraph.children]
        assert "strong" in element_types
        assert "emphasis" in element_types
        assert "codespan" in element_types

    def test_link_and_image(self) -> None:
        """Test parsing of links and images."""
        parser = MarkdownParser()
        markdown_text = "[link](http://a.com) ![alt](http://b.com/img.png)"
        root = parser.parse(markdown_text)

        paragraph = root.children[0]

        link = paragraph.children[0]
        assert link.type == "link"
        assert link.text == "link"
        assert link.metadata == {"url": "http://a.com"}

        image = paragraph.children[2]  # Index 1 is a space
        assert image.type == "image"
        assert image.text == "alt"
        assert image.metadata == {"src": "http://b.com/img.png"}
