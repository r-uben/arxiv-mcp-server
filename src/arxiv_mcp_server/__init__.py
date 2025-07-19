"""arXiv MCP Server - Search and retrieve academic papers from arXiv."""

__version__ = "0.1.0"

# Import main server class for easy access
from .server import ArxivMCPServer, create_server

# Re-export key classes from submodules
from .api import ArxivClient
from .utils import CitationFormatter, format_bibliography, NameNormalizer
from .extraction import PaperReader, PaperAnalyzer, SmartPDFExtractor, ExtractionTier
from .analysis import PaperComparator, CitationTracker

__all__ = [
    "ArxivMCPServer", 
    "create_server",
    "ArxivClient",
    "CitationFormatter", 
    "format_bibliography", 
    "NameNormalizer",
    "PaperReader", 
    "PaperAnalyzer", 
    "SmartPDFExtractor", 
    "ExtractionTier",
    "PaperComparator", 
    "CitationTracker"
]
