"""Tests for smart PDF extractor functionality."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from arxiv_mcp_server.smart_extractor import (
    DifficultyClassifier,
    ExtractionTier,
    NOUGATExtractor,
    GROBIDExtractor,
    MistralOCRClient,
    SmartPDFExtractor
)


class TestDifficultyClassifier:
    """Test PDF difficulty classification."""
    
    @pytest.fixture
    def classifier(self):
        return DifficultyClassifier()
    
    def test_math_density_calculation(self, classifier):
        """Test mathematical content density calculation."""
        # Text with math symbols
        math_text = "The equation $E = mc^2$ and ∫f(x)dx with Greek letters α, β, γ"
        density = classifier._calculate_math_density(math_text)
        assert density > 0.0
        
        # Text without math
        plain_text = "This is plain text without mathematical content"
        density = classifier._calculate_math_density(plain_text)
        assert density == 0.0
    
    def test_layout_complexity_calculation(self, classifier):
        """Test layout complexity score calculation."""
        # Complex layout text
        complex_text = "Figure 1 shows Table 2 with multi-column layout \\includegraphics{fig1}"
        complexity = classifier._calculate_layout_complexity(complex_text)
        assert complexity > 0.0
        
        # Simple layout text
        simple_text = "This is a simple paragraph of text"
        complexity = classifier._calculate_layout_complexity(simple_text)
        assert complexity == 0.0
    
    def test_text_extractability_calculation(self, classifier):
        """Test text extraction quality assessment."""
        # Good text
        good_text = "This is well-extracted text with proper formatting."
        quality = classifier._calculate_text_extractability(good_text)
        assert quality > 0.8
        
        # Poor extraction with artifacts
        poor_text = "This is bad�text with�many replacement chars"
        quality = classifier._calculate_text_extractability(poor_text)
        assert quality < 0.5
    
    def test_tier_determination(self, classifier):
        """Test tier determination logic."""
        # Simple document factors
        simple_factors = {
            "page_count": 5,
            "math_density": 0.01,
            "layout_complexity": 0.05,
            "text_extractability": 0.9,
            "file_size_mb": 2.0
        }
        tier, confidence, reasoning = classifier._determine_tier(simple_factors)
        assert tier == ExtractionTier.FAST
        
        # Complex document factors
        complex_factors = {
            "page_count": 60,
            "math_density": 0.15,
            "layout_complexity": 0.4,
            "text_extractability": 0.3,
            "file_size_mb": 15.0
        }
        tier, confidence, reasoning = classifier._determine_tier(complex_factors)
        assert tier == ExtractionTier.PREMIUM


class TestNOUGATExtractor:
    """Test NOUGAT extractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        return NOUGATExtractor()
    
    def test_section_parsing(self, extractor):
        """Test section parsing from NOUGAT markdown output."""
        content = """# Introduction
This is the introduction section.

## Methodology
This describes the methodology.

### Subsection
A subsection with details.

## Results
The results are presented here.
"""
        sections = extractor._parse_sections(content)
        
        assert "introduction" in sections
        assert "methodology" in sections
        assert "results" in sections
        assert "This is the introduction section." in sections["introduction"]
        assert "This describes the methodology." in sections["methodology"]


class TestGROBIDExtractor:
    """Test GROBID extractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        return GROBIDExtractor()
    
    def test_tei_xml_parsing(self, extractor):
        """Test TEI XML parsing."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
    <teiHeader>
        <titleStmt>
            <title type="main">Test Paper Title</title>
        </titleStmt>
        <sourceDesc>
            <biblStruct>
                <analytic>
                    <author>
                        <persName>
                            <forename type="first">John</forename>
                            <surname>Doe</surname>
                        </persName>
                    </author>
                </analytic>
            </biblStruct>
        </sourceDesc>
    </teiHeader>
    <text>
        <front>
            <abstract>
                <p>This is the abstract of the paper.</p>
            </abstract>
        </front>
        <body>
            <div>
                <head>Introduction</head>
                <p>This is the introduction section.</p>
            </div>
        </body>
    </text>
</TEI>"""
        
        parsed = extractor._parse_tei_xml(xml_content)
        
        assert "title" in parsed["metadata"]
        assert parsed["metadata"]["title"] == "Test Paper Title"
        assert "John Doe" in parsed["metadata"]["authors"]
        assert "abstract" in parsed["sections"]
        assert "introduction" in parsed["sections"]


class TestMistralOCRClient:
    """Test Mistral OCR client functionality."""
    
    @pytest.fixture
    def client(self):
        return MistralOCRClient(api_key="test-key")
    
    @patch('aiohttp.ClientSession.post')
    async def test_process_document_success(self, mock_post, client):
        """Test successful document processing."""
        # Mock response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "pages": [
                {
                    "index": 1,
                    "markdown": "# Test Paper\n\nThis is page 1 content.",
                    "images": [],
                    "dimensions": {"dpi": 150, "height": 800, "width": 600}
                },
                {
                    "index": 2,
                    "markdown": "## Results\n\nThis is page 2 content.",
                    "images": [],
                    "dimensions": {"dpi": 150, "height": 800, "width": 600}
                }
            ]
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create a test file path
        test_file = Path("/tmp/test.pdf")
        
        # Mock file operations
        with patch('builtins.open', mock_open(read_data=b"fake pdf content")):
            with patch('base64.b64encode', return_value=b"fake_base64"):
                client.session = Mock()
                result = await client.process_document(test_file)
        
        assert result["extraction_method"] == "mistral_ocr_v2"
        assert "# Page 1" in result["content"]
        assert "# Page 2" in result["content"]
        assert result["metadata"]["page_count"] == 2


class TestSmartPDFExtractor:
    """Test smart PDF extractor integration."""
    
    @pytest.fixture
    def extractor(self):
        return SmartPDFExtractor()
    
    @patch('arxiv_mcp_server.smart_extractor.NOUGATExtractor.extract_document')
    @patch('arxiv_mcp_server.smart_extractor.GROBIDExtractor.extract_document')
    async def test_smart_extraction_fallback(self, mock_grobid, mock_nougat, extractor):
        """Test SMART tier fallback mechanism."""
        # Setup mocks: NOUGAT fails, GROBID succeeds
        mock_nougat.side_effect = Exception("NOUGAT not available")
        mock_grobid.return_value = {
            "content": "Test content from GROBID",
            "extraction_method": "grobid_tei",
            "quality_estimate": 0.85
        }
        
        # Mock the basic extraction as final fallback
        with patch.object(extractor.paper_reader, 'download_and_read_paper') as mock_reader:
            with patch.object(extractor.paper_analyzer, 'summarize_paper') as mock_summarizer:
                with patch.object(extractor.paper_analyzer, 'extract_key_findings') as mock_findings:
                    mock_reader.return_value = {"content": {"clean_text": "fallback text"}}
                    mock_summarizer.return_value = "test summary"
                    mock_findings.return_value = "test findings"
                    
                    test_file = Path("/tmp/test.pdf")
                    result = await extractor._extract_smart(test_file)
                    
                    # Should get GROBID result, not fallback
                    assert result["content"] == "Test content from GROBID"
                    assert result["extraction_method"] == "grobid_tei"
    
    async def test_difficulty_analysis_integration(self, extractor):
        """Test difficulty analysis integration."""
        test_file = Path("/tmp/test.pdf")
        
        # Mock the classifier
        mock_analysis = {
            "tier_recommendation": ExtractionTier.SMART,
            "confidence": 0.8,
            "reasoning": ["Moderate complexity detected"],
            "factors": {
                "page_count": 10,
                "math_density": 0.1,
                "layout_complexity": 0.2,
                "text_extractability": 0.8,
                "file_size_mb": 5.0
            }
        }
        
        with patch.object(extractor.classifier, 'analyze_difficulty', return_value=mock_analysis):
            with patch.object(extractor, '_extract_with_tier', return_value={"test": "result"}):
                result = await extractor.extract_paper(test_file)
                
                assert result["tier_used"] == "smart"
                assert result["analysis"]["confidence"] == 0.8
                assert result["success"] == True


def mock_open(read_data=b""):
    """Mock file open for testing."""
    from unittest.mock import mock_open as base_mock_open
    return base_mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__])