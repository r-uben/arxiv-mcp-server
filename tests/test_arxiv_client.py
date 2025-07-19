"""Tests for the ArXiv client."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from arxiv_mcp_server.api.arxiv_client import ArxivClient, RateLimiter


class TestRateLimiter:
    """Test the rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        start_time = asyncio.get_event_loop().time()

        # Should allow 3 requests quickly
        for _ in range(3):
            await limiter.acquire()

        elapsed = asyncio.get_event_loop().time() - start_time
        assert elapsed < 0.1  # Should be very fast

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests exceeding the limit."""
        limiter = RateLimiter(max_requests=2, time_window=1.0)

        # First 2 requests should be fast
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be delayed
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time

        assert elapsed >= 0.9  # Should wait close to 1 second


class TestArxivClient:
    """Test the ArXiv client functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock()
        return session

    @pytest.fixture
    def client(self, mock_session):
        """Create an ArxivClient with mocked session."""
        return ArxivClient(session=mock_session)

    def test_extract_arxiv_id(self, client):
        """Test arXiv ID extraction from URLs."""
        test_cases = [
            ("http://arxiv.org/abs/2301.00001v1", "2301.00001v1"),
            ("https://arxiv.org/abs/1234.5678", "1234.5678"),
            ("2301.00001", "2301.00001"),  # Should return as-is if no URL
        ]

        for url, expected_id in test_cases:
            result = client._extract_arxiv_id(url)
            assert result == expected_id

    def test_clean_text(self, client):
        """Test text cleaning functionality."""
        test_cases = [
            ("  Multiple   spaces  ", "Multiple spaces"),
            ("Text\nwith\nnewlines", "Text with newlines"),
            ("LaTeX $symbols$ removed", "LaTeX symbols removed"),
            ("Normal text", "Normal text"),
        ]

        for input_text, expected in test_cases:
            result = client._clean_text(input_text)
            assert result == expected

    @pytest.mark.asyncio
    async def test_get_paper_details_empty_id(self, client):
        """Test that empty arXiv ID raises ValueError."""
        with pytest.raises(ValueError, match="arXiv ID cannot be empty"):
            await client.get_paper_details("")

    @pytest.mark.asyncio
    async def test_parse_arxiv_response_with_sample_xml(self, client):
        """Test parsing of arXiv XML response."""
        sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2301.00001v1</id>
            <title>Test Paper Title</title>
            <summary>This is a test abstract with some content.</summary>
            <published>2023-01-01T00:00:00Z</published>
            <updated>2023-01-01T00:00:00Z</updated>
            <author>
              <name>John Doe</name>
            </author>
            <author>
              <name>Jane Smith</name>
            </author>
            <category term="cs.AI" />
            <category term="cs.LG" />
            <link rel="alternate" href="http://arxiv.org/abs/2301.00001v1" />
            <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1.pdf" />
            <arxiv:comment>10 pages, 5 figures</arxiv:comment>
            <arxiv:doi>10.1000/test</arxiv:doi>
          </entry>
        </feed>"""

        papers = client._parse_arxiv_response(sample_xml)

        assert len(papers) == 1
        paper = papers[0]

        assert paper["id"] == "2301.00001v1"
        assert paper["title"] == "Test Paper Title"
        assert paper["abstract"] == "This is a test abstract with some content."
        assert paper["authors"] == ["John Doe", "Jane Smith"]
        assert paper["categories"] == ["cs.AI", "cs.LG"]
        assert paper["url"] == "http://arxiv.org/abs/2301.00001v1"
        assert paper["pdf_url"] == "http://arxiv.org/pdf/2301.00001v1.pdf"
        assert paper["comment"] == "10 pages, 5 figures"
        assert paper["doi"] == "10.1000/test"

    def test_filter_by_date(self, client):
        """Test date filtering functionality."""
        papers = [
            {"id": "1", "published": "2023-01-01T00:00:00Z"},
            {"id": "2", "published": "2023-06-15T00:00:00Z"},
            {"id": "3", "published": "2023-12-31T00:00:00Z"},
        ]

        # Filter by start date
        filtered = client._filter_by_date(papers, start_date="2023-06-01")
        assert len(filtered) == 2
        assert filtered[0]["id"] == "2"
        assert filtered[1]["id"] == "3"

        # Filter by end date
        filtered = client._filter_by_date(papers, end_date="2023-06-30")
        assert len(filtered) == 2
        assert filtered[0]["id"] == "1"
        assert filtered[1]["id"] == "2"

        # Filter by date range
        filtered = client._filter_by_date(
            papers, start_date="2023-02-01", end_date="2023-11-30"
        )
        assert len(filtered) == 1
        assert filtered[0]["id"] == "2"


@pytest.mark.asyncio
async def test_context_manager():
    """Test that ArxivClient works as a context manager."""
    async with ArxivClient() as client:
        assert client.session is not None

    # Session should be closed after exiting context
