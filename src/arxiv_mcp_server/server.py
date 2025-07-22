"""Main MCP server implementation for arXiv paper search."""

import logging
from typing import Any, Dict, List

from mcp.server.models import InitializationOptions
from mcp.server import Server
from mcp.types import Tool, TextContent, ServerCapabilities, ToolsCapability

from .api.arxiv_client import ArxivClient
from .utils.citation_utils import CitationFormatter, format_bibliography
from .extraction.paper_reader import PaperReader, PaperAnalyzer
from .analysis.paper_analysis import PaperComparator, CitationTracker
from .extraction.smart_extractor import SmartPDFExtractor, ExtractionTier

logger = logging.getLogger(__name__)


class ArxivMCPServer:
    """MCP Server for arXiv paper search and retrieval."""

    def __init__(self):
        self.server = Server("arxiv-mcp-server")
        self.arxiv_client = ArxivClient()
        self.paper_reader = PaperReader()
        self.paper_analyzer = PaperAnalyzer()
        self.paper_comparator = PaperComparator()
        self.citation_tracker = CitationTracker()
        self.smart_extractor = SmartPDFExtractor()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP request handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="search_papers",
                    description="Search for academic papers on arXiv with smart author name handling",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords, titles, authors). For authors, will automatically try variations with/without accents and different hyphenation.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10, max: 100)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by arXiv categories (e.g., cs.AI, math.CO)",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date filter (YYYY-MM-DD)",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date filter (YYYY-MM-DD)",
                            },
                            "smart_author_search": {
                                "type": "boolean",
                                "description": "Enable smart author name variations (default: true)",
                                "default": True,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_paper_details",
                    description="Get detailed information about a specific paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID (e.g., 2301.00001 or 1234.5678v2)",
                            }
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="get_recent_papers",
                    description="Get recent papers from specific arXiv categories",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "arXiv category (e.g., cs.AI, math.CO, physics.gen-ph)",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10, max: 50)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50,
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days to look back (default: 7)",
                                "default": 7,
                                "minimum": 1,
                                "maximum": 30,
                            },
                        },
                        "required": ["category"],
                    },
                ),
                Tool(
                    name="get_author_papers",
                    description="Get all papers by a specific author with smart name matching",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "author_name": {
                                "type": "string",
                                "description": "Author name to search for (will try variations automatically)",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 20, max: 100)",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filter by arXiv categories (e.g., cs.AI, math.CO)",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date filter (YYYY-MM-DD)",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date filter (YYYY-MM-DD)",
                            },
                        },
                        "required": ["author_name"],
                    },
                ),
                Tool(
                    name="find_similar_papers",
                    description="Find papers similar to a reference paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "reference_paper_id": {
                                "type": "string",
                                "description": "arXiv ID of the reference paper",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of similar papers (default: 10, max: 20)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "similarity_method": {
                                "type": "string",
                                "description": "Method for finding similarity: keywords, categories, or authors",
                                "enum": ["keywords", "categories", "authors"],
                                "default": "keywords",
                            },
                        },
                        "required": ["reference_paper_id"],
                    },
                ),
                Tool(
                    name="format_citation",
                    description="Format a paper citation in various academic styles",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to format",
                            },
                            "style": {
                                "type": "string",
                                "description": "Citation style: APA, MLA, Chicago, or BibTeX",
                                "enum": ["apa", "mla", "chicago", "bibtex"],
                                "default": "apa",
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="export_bibliography",
                    description="Export multiple papers as a formatted bibliography",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of arXiv paper IDs to include in bibliography",
                            },
                            "style": {
                                "type": "string",
                                "description": "Citation style: APA, MLA, Chicago, or BibTeX",
                                "enum": ["apa", "mla", "chicago", "bibtex"],
                                "default": "apa",
                            },
                        },
                        "required": ["arxiv_ids"],
                    },
                ),
                Tool(
                    name="download_and_read_paper",
                    description="Download and extract full text content from an ArXiv paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to download and read",
                            },
                            "format_type": {
                                "type": "string",
                                "description": "Format to download: pdf or tex",
                                "enum": ["pdf", "tex"],
                                "default": "pdf",
                            },
                            "force_download": {
                                "type": "boolean",
                                "description": "Force re-download even if cached",
                                "default": False,
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="summarize_paper",
                    description="Generate a structured summary of a paper with key sections",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to summarize",
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="extract_key_findings",
                    description="Extract key findings, contributions, and methodology from a paper",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to analyze",
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="compare_papers",
                    description="Compare multiple papers across different aspects",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of arXiv paper IDs to compare",
                                "minItems": 2,
                                "maxItems": 5,
                            },
                            "comparison_aspects": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["methodology", "results", "contributions", "scope"]
                                },
                                "description": "Aspects to compare (default: all)",
                                "default": ["methodology", "results", "contributions", "scope"],
                            },
                        },
                        "required": ["paper_ids"],
                    },
                ),
                Tool(
                    name="find_citing_papers",
                    description="Find papers that cite a given ArXiv paper using Semantic Scholar",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to find citations for",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of citing papers (default: 20, max: 50)",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 50,
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="get_citation_network",
                    description="Build a citation network around a paper showing references and citations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to build network around",
                            },
                            "depth": {
                                "type": "integer",
                                "description": "Network depth (default: 2)",
                                "default": 2,
                                "minimum": 1,
                                "maximum": 3,
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="smart_extract_paper",
                    description="Advanced PDF extraction with three-tier adaptive mechanism (FAST/SMART/PREMIUM)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to extract content from",
                            },
                            "extraction_tier": {
                                "type": "string",
                                "description": "Force specific extraction tier: fast (pdfplumber), smart (NOUGAT/GROBID), premium (Mistral OCR)",
                                "enum": ["fast", "smart", "premium"],
                            },
                            "budget_mode": {
                                "type": "boolean",
                                "description": "If true, avoid paid services (no premium tier)",
                                "default": False,
                            },
                            "force_analysis": {
                                "type": "boolean", 
                                "description": "Always analyze difficulty even with user-specified tier",
                                "default": False,
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
                Tool(
                    name="analyze_paper_difficulty",
                    description="Analyze PDF complexity to determine optimal extraction method",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "arXiv paper ID to analyze",
                            },
                        },
                        "required": ["arxiv_id"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: Dict[str, Any]
        ) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "search_papers":
                    results = await self.arxiv_client.search_papers(
                        query=arguments["query"],
                        max_results=arguments.get("max_results", 10),
                        categories=arguments.get("categories"),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date"),
                        smart_author_search=arguments.get("smart_author_search", True),
                    )
                    return [
                        TextContent(
                            type="text", text=self._format_search_results(results)
                        )
                    ]

                elif name == "get_paper_details":
                    paper = await self.arxiv_client.get_paper_details(
                        arguments["arxiv_id"]
                    )
                    return [
                        TextContent(type="text", text=self._format_paper_details(paper))
                    ]

                elif name == "get_recent_papers":
                    papers = await self.arxiv_client.get_recent_papers(
                        category=arguments["category"],
                        max_results=arguments.get("max_results", 10),
                        days_back=arguments.get("days_back", 7),
                    )
                    return [
                        TextContent(
                            type="text", text=self._format_search_results(papers)
                        )
                    ]

                elif name == "get_author_papers":
                    papers = await self.arxiv_client.get_author_papers(
                        author_name=arguments["author_name"],
                        max_results=arguments.get("max_results", 20),
                        categories=arguments.get("categories"),
                        start_date=arguments.get("start_date"),
                        end_date=arguments.get("end_date"),
                    )
                    return [
                        TextContent(
                            type="text", text=self._format_search_results(papers)
                        )
                    ]

                elif name == "find_similar_papers":
                    papers = await self.arxiv_client.find_similar_papers(
                        reference_paper_id=arguments["reference_paper_id"],
                        max_results=arguments.get("max_results", 10),
                        similarity_method=arguments.get("similarity_method", "keywords"),
                    )
                    return [
                        TextContent(
                            type="text", text=self._format_search_results(papers)
                        )
                    ]

                elif name == "format_citation":
                    paper = await self.arxiv_client.get_paper_details(
                        arguments["arxiv_id"]
                    )
                    if not paper:
                        raise ValueError(f"Paper {arguments['arxiv_id']} not found")
                    
                    formatter = CitationFormatter()
                    citation = formatter.format_citation(
                        paper, 
                        style=arguments.get("style", "apa")
                    )
                    return [TextContent(type="text", text=citation)]

                elif name == "export_bibliography":
                    papers = []
                    for arxiv_id in arguments["arxiv_ids"]:
                        paper = await self.arxiv_client.get_paper_details(arxiv_id)
                        if paper:
                            papers.append(paper)
                    
                    bibliography = format_bibliography(
                        papers, 
                        style=arguments.get("style", "apa")
                    )
                    return [TextContent(type="text", text=bibliography)]

                elif name == "download_and_read_paper":
                    result = await self.paper_reader.download_and_read_paper(
                        arxiv_id=arguments["arxiv_id"],
                        format_type=arguments.get("format_type", "pdf"),
                        force_download=arguments.get("force_download", False),
                    )
                    return [TextContent(type="text", text=self._format_paper_content(result))]

                elif name == "summarize_paper":
                    summary = await self.paper_analyzer.summarize_paper(
                        arguments["arxiv_id"]
                    )
                    return [TextContent(type="text", text=self._format_paper_summary(summary))]

                elif name == "extract_key_findings":
                    findings = await self.paper_analyzer.extract_key_findings(
                        arguments["arxiv_id"]
                    )
                    return [TextContent(type="text", text=self._format_key_findings(findings))]

                elif name == "compare_papers":
                    comparison = await self.paper_comparator.compare_papers(
                        paper_ids=arguments["paper_ids"],
                        comparison_aspects=arguments.get("comparison_aspects"),
                    )
                    return [TextContent(type="text", text=self._format_paper_comparison(comparison))]

                elif name == "find_citing_papers":
                    citing_papers = await self.citation_tracker.find_citing_papers(
                        arxiv_id=arguments["arxiv_id"],
                        max_results=arguments.get("max_results", 20),
                    )
                    return [TextContent(type="text", text=self._format_citing_papers(citing_papers))]

                elif name == "get_citation_network":
                    network = await self.citation_tracker.get_citation_network(
                        arxiv_id=arguments["arxiv_id"],
                        depth=arguments.get("depth", 2),
                    )
                    return [TextContent(type="text", text=self._format_citation_network(network))]

                elif name == "smart_extract_paper":
                    # Download paper first if not cached
                    await self.paper_reader.download_and_read_paper(
                        arxiv_id=arguments["arxiv_id"],
                        format_type="pdf"
                    )
                    
                    # Get the cached PDF path
                    pdf_path = self.paper_reader.cache_dir / f"{arguments['arxiv_id']}.pdf"
                    
                    # Parse extraction tier if provided
                    user_preference = None
                    if arguments.get("extraction_tier"):
                        user_preference = ExtractionTier(arguments["extraction_tier"])
                    
                    result = await self.smart_extractor.extract_paper(
                        pdf_path=pdf_path,
                        user_preference=user_preference,
                        budget_mode=arguments.get("budget_mode", False),
                        force_analysis=arguments.get("force_analysis", False),
                    )
                    return [TextContent(type="text", text=self._format_smart_extraction(result))]

                elif name == "analyze_paper_difficulty":
                    # Download paper first if not cached
                    await self.paper_reader.download_and_read_paper(
                        arxiv_id=arguments["arxiv_id"],
                        format_type="pdf"
                    )
                    
                    # Get the cached PDF path
                    pdf_path = self.paper_reader.cache_dir / f"{arguments['arxiv_id']}.pdf"
                    
                    analysis = await self.smart_extractor.classifier.analyze_difficulty(pdf_path)
                    return [TextContent(type="text", text=self._format_difficulty_analysis(analysis))]

                else:
                    raise ValueError(f"Unknown tool: {name}")

            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _format_search_results(self, papers: List[Dict[str, Any]]) -> str:
        """Format search results for display."""
        if not papers:
            return "No papers found matching your query."

        result = f"Found {len(papers)} paper(s):\n\n"

        for i, paper in enumerate(papers, 1):
            result += f"**{i}. {paper['title']}**\n"
            result += f"Authors: {', '.join(paper['authors'])}\n"
            result += f"arXiv ID: {paper['id']}\n"
            result += f"Published: {paper['published']}\n"
            result += f"Categories: {', '.join(paper['categories'])}\n"
            result += f"Abstract: {paper['abstract'][:200]}...\n"
            result += f"URL: {paper['url']}\n"
            result += f"PDF: {paper['pdf_url']}\n\n"

        return result

    def _format_paper_details(self, paper: Dict[str, Any]) -> str:
        """Format detailed paper information."""
        if not paper:
            return "Paper not found."

        result = f"**{paper['title']}**\n\n"
        result += f"**Authors:** {', '.join(paper['authors'])}\n"
        result += f"**arXiv ID:** {paper['id']}\n"
        result += f"**Published:** {paper['published']}\n"
        result += f"**Updated:** {paper['updated']}\n"
        result += f"**Categories:** {', '.join(paper['categories'])}\n"
        result += f"**URL:** {paper['url']}\n"
        result += f"**PDF:** {paper['pdf_url']}\n\n"
        result += f"**Abstract:**\n{paper['abstract']}\n"

        if paper.get("comment"):
            result += f"\n**Comment:** {paper['comment']}\n"

        if paper.get("journal_ref"):
            result += f"**Journal Reference:** {paper['journal_ref']}\n"

        if paper.get("doi"):
            result += f"**DOI:** {paper['doi']}\n"

        return result

    def _format_paper_content(self, result: Dict[str, Any]) -> str:
        """Format paper content extraction results."""
        if "error" in result:
            return f"Error: {result['error']}"
        
        content = result["content"]
        output = f"# Paper Content: {result['arxiv_id']}\n\n"
        
        output += f"**Format:** {result['format']}\n"
        output += f"**Pages:** {content.get('page_count', 'Unknown')}\n"
        output += f"**Word Count:** {content.get('word_count', 'Unknown')}\n"
        output += f"**Extraction Method:** {content.get('extraction_method', 'Unknown')}\n\n"
        
        if content.get("sections"):
            output += "## Identified Sections:\n"
            for section_name, section_text in content["sections"].items():
                output += f"### {section_name.title()}\n"
                preview = section_text[:500] + "..." if len(section_text) > 500 else section_text
                output += f"{preview}\n\n"
        
        if result.get("cached_path"):
            output += f"**Cached at:** {result['cached_path']}\n"
        
        return output

    def _format_paper_summary(self, summary: Dict[str, Any]) -> str:
        """Format paper summary results."""
        if "error" in summary:
            return f"Error: {summary['error']}"
        
        output = f"# Paper Summary: {summary['arxiv_id']}\n\n"
        
        metadata = summary.get("metadata", {})
        output += f"**Pages:** {metadata.get('page_count', 'Unknown')}\n"
        output += f"**Words:** {metadata.get('word_count', 'Unknown')}\n"
        output += f"**Sections Found:** {', '.join(summary.get('sections_found', []))}\n\n"
        
        if summary.get("abstract") and summary["abstract"] != "Not found":
            output += "## Abstract\n"
            output += f"{summary['abstract']}\n\n"
        
        if summary.get("key_sections"):
            output += "## Key Sections\n"
            for section_name, section_info in summary["key_sections"].items():
                output += f"### {section_name.title()}\n"
                output += f"**Length:** {section_info['length']} words\n"
                output += f"**Preview:** {section_info['preview']}\n\n"
        
        if summary.get("overview"):
            output += "## Overview\n"
            output += f"{summary['overview']}\n"
        
        return output

    def _format_key_findings(self, findings: Dict[str, Any]) -> str:
        """Format key findings extraction results."""
        if "error" in findings:
            return f"Error: {findings['error']}"
        
        output = f"# Key Findings: {findings['arxiv_id']}\n\n"
        
        if findings.get("contributions"):
            output += "## Main Contributions\n"
            for i, contrib in enumerate(findings["contributions"], 1):
                output += f"{i}. {contrib}\n"
            output += "\n"
        
        if findings.get("results"):
            output += "## Key Results\n"
            for i, result in enumerate(findings["results"], 1):
                output += f"{i}. {result}\n"
            output += "\n"
        
        if findings.get("methodology"):
            method = findings["methodology"]
            output += "## Methodology\n"
            if isinstance(method, dict):
                output += f"**Approach:** {method.get('approach', 'Not available')}\n"
                output += f"**Complexity:** {method.get('length', 0)} words\n"
                output += f"**Algorithms Mentioned:** {method.get('algorithms_mentioned', 0)}\n\n"
            else:
                output += f"{method}\n\n"
        
        if findings.get("conclusions"):
            output += "## Conclusions\n"
            for i, conclusion in enumerate(findings["conclusions"], 1):
                output += f"{i}. {conclusion}\n"
            output += "\n"
        
        if findings.get("equations"):
            output += "## Mathematical Content\n"
            output += f"Found {len(findings['equations'])} equations/formulas\n"
            for i, eq in enumerate(findings["equations"][:3], 1):  # Show first 3
                output += f"{i}. `{eq[:100]}{'...' if len(eq) > 100 else ''}`\n"
            output += "\n"
        
        if findings.get("figures_mentioned"):
            output += "## Figures Referenced\n"
            for i, fig_ref in enumerate(findings["figures_mentioned"], 1):
                output += f"{i}. {fig_ref}\n"
        
        return output

    def _format_paper_comparison(self, comparison: Dict[str, Any]) -> str:
        """Format paper comparison results."""
        if "error" in comparison:
            return f"Error: {comparison['error']}"
        
        output = f"# Paper Comparison\n\n"
        output += f"**Papers Analyzed:** {len(comparison['papers_analyzed'])}\n"
        output += f"**Comparison Aspects:** {', '.join(comparison['comparison_aspects'])}\n\n"
        
        if comparison.get("overview"):
            overview = comparison["overview"]
            output += "## Overview\n"
            output += f"**Average Pages:** {overview.get('avg_page_count', 0):.1f}\n"
            output += f"**Average Words:** {overview.get('avg_word_count', 0):.0f}\n"
            if overview.get("common_sections"):
                output += f"**Common Sections:** {', '.join(overview['common_sections'])}\n"
            output += "\n"
        
        if comparison.get("similarities"):
            output += "## Similarities\n"
            for similarity in comparison["similarities"]:
                output += f"- {similarity}\n"
            output += "\n"
        
        if comparison.get("differences"):
            output += "## Key Differences\n"
            for difference in comparison["differences"]:
                output += f"- {difference}\n"
            output += "\n"
        
        if comparison.get("complementary_aspects"):
            output += "## Complementary Aspects\n"
            for aspect in comparison["complementary_aspects"]:
                output += f"- {aspect}\n"
            output += "\n"
        
        # Detailed comparison sections
        detailed = comparison.get("detailed_comparison", {})
        for aspect_name, aspect_data in detailed.items():
            output += f"## {aspect_name.title()} Comparison\n"
            if isinstance(aspect_data, dict):
                for key, value in aspect_data.items():
                    if isinstance(value, dict):
                        output += f"**{key.replace('_', ' ').title()}:**\n"
                        for subkey, subvalue in value.items():
                            output += f"  - {subkey}: {subvalue}\n"
                    elif isinstance(value, list):
                        output += f"**{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}\n"
                    else:
                        output += f"**{key.replace('_', ' ').title()}:** {value}\n"
                output += "\n"
            else:
                output += f"{aspect_data}\n\n"
        
        return output

    def _format_citing_papers(self, citing_papers: Dict[str, Any]) -> str:
        """Format citing papers results."""
        if "error" in citing_papers:
            return f"Error: {citing_papers['error']}"
        
        output = f"# Citation Analysis: {citing_papers['arxiv_id']}\n\n"
        output += f"**Total Citations Found:** {citing_papers['citation_count']}\n"
        output += f"**Semantic Scholar ID:** {citing_papers.get('s2_paper_id', 'Unknown')}\n\n"
        
        if citing_papers.get("most_recent_citations"):
            output += "## Most Recent Citations\n"
            for i, paper in enumerate(citing_papers["most_recent_citations"], 1):
                title = paper.get("title", "Unknown Title")
                year = paper.get("year", "Unknown")
                authors = paper.get("authors", [])
                author_names = [a.get("name", "") for a in authors[:3]]
                
                output += f"{i}. **{title}** ({year})\n"
                if author_names:
                    output += f"   Authors: {', '.join(author_names)}{'...' if len(authors) > 3 else ''}\n"
                if paper.get("citationCount"):
                    output += f"   Citations: {paper['citationCount']}\n"
                
                arxiv_id = paper.get("externalIds", {}).get("ArXiv")
                if arxiv_id:
                    output += f"   ArXiv: {arxiv_id}\n"
                output += "\n"
        
        if citing_papers.get("citing_papers"):
            remaining = len(citing_papers["citing_papers"]) - len(citing_papers.get("most_recent_citations", []))
            if remaining > 0:
                output += f"## Additional {remaining} citing papers available in full results\n"
        
        return output

    def _format_citation_network(self, network: Dict[str, Any]) -> str:
        """Format citation network results."""
        if "error" in network:
            return f"Error: {network['error']}"
        
        output = f"# Citation Network: {network['center_paper']}\n\n"
        
        stats = network.get("statistics", {})
        output += f"**Network Depth:** {network['depth']}\n"
        output += f"**Total Nodes:** {stats.get('total_nodes', 0)}\n"
        output += f"**Total Edges:** {stats.get('total_edges', 0)}\n"
        output += f"**References:** {stats.get('references_count', 0)}\n"
        output += f"**Citations:** {stats.get('citations_count', 0)}\n"
        output += f"**ArXiv Papers in Network:** {stats.get('arxiv_papers_in_network', 0)}\n\n"
        
        if network.get("edges"):
            output += "## Citation Relationships\n"
            
            # Group edges by type
            cites_edges = [e for e in network["edges"] if e.get("type") == "cites"]
            
            if cites_edges:
                output += "### Citation Flow\n"
                for edge in cites_edges[:10]:  # Show first 10
                    from_paper = edge["from"]
                    to_paper = edge["to"]
                    
                    # Get paper titles if available
                    from_title = network.get("nodes", {}).get(from_paper, {}).get("title", from_paper)
                    to_title = network.get("nodes", {}).get(to_paper, {}).get("title", to_paper)
                    
                    output += f"- {from_title[:50]}{'...' if len(from_title) > 50 else ''}\n"
                    output += f"  → cites → {to_title[:50]}{'...' if len(to_title) > 50 else ''}\n\n"
        
        if network.get("nodes"):
            arxiv_nodes = {k: v for k, v in network["nodes"].items() if k != network["center_paper"]}
            if arxiv_nodes:
                output += f"## Related ArXiv Papers ({len(arxiv_nodes)})\n"
                for arxiv_id, paper_data in list(arxiv_nodes.items())[:5]:  # Show first 5
                    title = paper_data.get("title", arxiv_id)
                    year = paper_data.get("year", "Unknown")
                    citations = paper_data.get("citationCount", 0)
                    
                    output += f"- **{title}** ({year})\n"
                    output += f"  ArXiv: {arxiv_id} | Citations: {citations}\n\n"
        
        return output

    def _format_smart_extraction(self, result: Dict[str, Any]) -> str:
        """Format smart PDF extraction results."""
        if not result.get("success"):
            return f"Error in smart extraction: {result.get('extraction', {}).get('error', 'Unknown error')}"
        
        analysis = result.get("analysis", {})
        extraction = result.get("extraction", {})
        
        output = f"# Smart PDF Extraction Results\n\n"
        output += f"**Paper:** {result.get('pdf_path', '').split('/')[-1]}\n"
        output += f"**Extraction Tier:** {result.get('tier_used', 'unknown').upper()}\n"
        output += f"**Success:** {'✅' if result.get('success') else '❌'}\n\n"
        
        # Analysis details
        output += "## Difficulty Analysis\n"
        output += f"**Recommended Tier:** {analysis.get('tier_recommendation', 'unknown').value.upper()}\n"
        output += f"**Confidence:** {analysis.get('confidence', 0):.1%}\n"
        
        factors = analysis.get("factors", {})
        if factors:
            output += f"**File Size:** {factors.get('file_size_mb', 0):.1f} MB\n"
            output += f"**Page Count:** {factors.get('page_count', 0)}\n"
            output += f"**Math Density:** {factors.get('math_density', 0):.1%}\n"
            output += f"**Layout Complexity:** {factors.get('layout_complexity', 0):.1%}\n"
            output += f"**Text Extractability:** {factors.get('text_extractability', 0):.1%}\n"
        
        reasoning = analysis.get("reasoning", [])
        if reasoning:
            output += f"**Reasoning:** {'; '.join(reasoning)}\n"
        
        output += "\n"
        
        # Extraction results
        output += "## Extraction Results\n"
        output += f"**Method:** {extraction.get('extraction_method', 'unknown')}\n"
        output += f"**Processing Time:** {extraction.get('processing_time', 'unknown')}\n"
        output += f"**Quality Estimate:** {extraction.get('quality_estimate', 0):.1%}\n"
        
        # Add fallback information if present
        if extraction.get("fallback_info"):
            fallback_info = extraction["fallback_info"]
            if fallback_info.get("user_message"):
                output += f"\n**Note:** {fallback_info['user_message']}\n"
        
        if extraction.get("processing_note"):
            output += f"**Processing:** {extraction['processing_note']}\n"
            
        output += "\n"
        
        # Content preview
        content = extraction.get("content", "")
        if content:
            output += "## Content Preview\n"
            preview = content[:500] + "..." if len(content) > 500 else content
            output += f"{preview}\n\n"
        
        # Special features for different tiers
        if extraction.get("tables"):
            output += f"## Tables Extracted\n"
            output += f"Found {len(extraction['tables'])} tables\n\n"
        
        if extraction.get("equations"):
            output += f"## Equations Extracted\n"
            output += f"Found {len(extraction['equations'])} equations\n"
            if extraction["equations"]:
                output += f"Example: {extraction['equations'][0][:100]}...\n\n"
        
        if extraction.get("figures"):
            output += f"## Figures Detected\n"
            output += f"Found {len(extraction['figures'])} figures\n\n"
        
        # Sections if available
        sections = extraction.get("sections", {})
        if sections:
            output += "## Identified Sections\n"
            for section_name, section_text in sections.items():
                output += f"### {section_name.title()}\n"
                section_preview = section_text[:200] + "..." if len(section_text) > 200 else section_text
                output += f"{section_preview}\n\n"
        
        return output

    def _format_difficulty_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format PDF difficulty analysis results."""
        output = f"# PDF Difficulty Analysis\n\n"
        
        output += f"**Recommended Tier:** {analysis.get('tier_recommendation', 'unknown').value.upper()}\n"
        output += f"**Confidence:** {analysis.get('confidence', 0):.1%}\n\n"
        
        factors = analysis.get("factors", {})
        if factors:
            output += "## Analysis Factors\n"
            output += f"**File Size:** {factors.get('file_size_mb', 0):.1f} MB\n"
            output += f"**Page Count:** {factors.get('page_count', 0)}\n"
            output += f"**Math Density:** {factors.get('math_density', 0):.1%}\n"
            output += f"**Layout Complexity:** {factors.get('layout_complexity', 0):.1%}\n"
            output += f"**Text Extractability:** {factors.get('text_extractability', 0):.1%}\n\n"
        
        reasoning = analysis.get("reasoning", [])
        if reasoning:
            output += "## Reasoning\n"
            for i, reason in enumerate(reasoning, 1):
                output += f"{i}. {reason}\n"
            output += "\n"
        
        # Tier recommendations
        output += "## Tier Descriptions\n"
        output += "**FAST:** pdfplumber + PyPDF2 (simple papers, ~1s)\n"
        output += "**SMART:** NOUGAT/GROBID enhanced (moderate complexity, ~5s)\n"
        output += "**PREMIUM:** Mistral OCR (math/complex layouts, ~2s, 94.89% accuracy)\n\n"
        
        # Cost implications
        tier = analysis.get('tier_recommendation', ExtractionTier.FAST)
        if tier == ExtractionTier.PREMIUM:
            output += "## Cost Note\n"
            output += "⚠️ PREMIUM tier uses Mistral OCR API (paid service)\n"
            output += "Use budget_mode=true to avoid paid services\n"
        
        return output

    async def run(self, transport_type: str = "stdio"):
        """Run the MCP server."""
        if transport_type == "stdio":
            from mcp.server.stdio import stdio_server

            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name="arxiv-mcp-server",
                        server_version="0.1.0",
                        capabilities=ServerCapabilities(
                            tools=ToolsCapability(),
                        ),
                    ),
                )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")


def create_server() -> ArxivMCPServer:
    """Create and return an ArxivMCPServer instance."""
    return ArxivMCPServer()


def main():
    """Main entry point for the MCP server."""
    import asyncio
    import os
    
    # Set FORCE_SMART for optimal academic paper extraction
    os.environ["FORCE_SMART"] = "true"
    logger.info("Setting FORCE_SMART=true for optimal academic paper extraction")
    
    logger.info("Starting arXiv MCP Server...")
    
    server = create_server()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
