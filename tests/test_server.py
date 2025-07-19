"""Tests for the MCP server."""

import pytest

from arxiv_mcp_server.server import ArxivMCPServer


class TestArxivMCPServer:
    """Test the MCP server functionality."""

    @pytest.fixture
    def server(self):
        """Create an ArxivMCPServer instance."""
        return ArxivMCPServer()

    def test_server_initialization(self, server):
        """Test that server initializes correctly."""
        assert server.server is not None
        assert server.arxiv_client is not None
        assert server.server.name == "arxiv-mcp-server"

    def test_format_search_results_empty(self, server):
        """Test formatting of empty search results."""
        result = server._format_search_results([])
        assert "No papers found" in result

    def test_format_search_results_with_papers(self, server):
        """Test formatting of search results with papers."""
        papers = [
            {
                "id": "2301.00001",
                "title": "Test Paper",
                "authors": ["John Doe", "Jane Smith"],
                "published": "2023-01-01T00:00:00Z",
                "categories": ["cs.AI"],
                "abstract": "This is a test abstract that is longer than 200 characters and should be truncated when displayed in the search results to keep the output manageable.",
                "url": "http://arxiv.org/abs/2301.00001",
                "pdf_url": "http://arxiv.org/pdf/2301.00001.pdf",
            }
        ]

        result = server._format_search_results(papers)

        assert "Found 1 paper(s)" in result
        assert "Test Paper" in result
        assert "John Doe, Jane Smith" in result
        assert "2301.00001" in result
        assert "cs.AI" in result
        assert "..." in result  # Abstract should be truncated

    def test_format_paper_details_empty(self, server):
        """Test formatting of empty paper details."""
        result = server._format_paper_details(None)
        assert "Paper not found" in result

    def test_format_paper_details_with_paper(self, server):
        """Test formatting of detailed paper information."""
        paper = {
            "id": "2301.00001",
            "title": "Test Paper",
            "authors": ["John Doe", "Jane Smith"],
            "published": "2023-01-01T00:00:00Z",
            "updated": "2023-01-02T00:00:00Z",
            "categories": ["cs.AI", "cs.LG"],
            "abstract": "This is a complete test abstract.",
            "url": "http://arxiv.org/abs/2301.00001",
            "pdf_url": "http://arxiv.org/pdf/2301.00001.pdf",
            "comment": "10 pages, 5 figures",
            "journal_ref": "Nature 2023",
            "doi": "10.1000/test",
        }

        result = server._format_paper_details(paper)

        assert "Test Paper" in result
        assert "John Doe, Jane Smith" in result
        assert "2301.00001" in result
        assert "cs.AI, cs.LG" in result
        assert "This is a complete test abstract." in result
        assert "10 pages, 5 figures" in result
        assert "Nature 2023" in result
        assert "10.1000/test" in result


# Note: MCP Server handler testing is complex due to internal architecture
# These tests focus on the core functionality that can be tested directly
