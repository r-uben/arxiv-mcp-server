# arXiv MCP Server

A Model Context Protocol (MCP) server for searching and retrieving academic papers from arXiv. This server enables Claude Code and other MCP-compatible tools to search arXiv papers, get detailed paper information, and retrieve recent papers by category.

## Features

### Core Search & Discovery
- **Search papers**: Search arXiv by keywords, authors, categories, and date ranges
- **Paper details**: Get detailed information about specific papers by arXiv ID
- **Recent papers**: Retrieve recent papers from specific arXiv categories
- **Author search**: Find all papers by specific authors with smart name matching
- **Similar papers**: Discover related papers using keywords, categories, or authors

### Paper Content Analysis
- **Full paper reading**: Download and extract complete text from PDF files
- **Three-tier smart extraction**: Adaptive PDF processing (FAST/SMART/PREMIUM)
- **Intelligent summarization**: Generate structured summaries with key sections
- **Key findings extraction**: Identify contributions, methodology, and results
- **Multi-paper comparison**: Compare papers across methodology, results, and scope
- **Mathematical content**: Extract equations and mathematical formulations
- **PDF difficulty analysis**: Automatic complexity assessment and tier recommendation

### Citation & Bibliography Tools
- **Citation formatting**: Generate properly formatted citations (APA, MLA, Chicago, BibTeX)
- **Bibliography export**: Create formatted bibliographies for multiple papers
- **Citation tracking**: Find papers that cite a given work (via Semantic Scholar)
- **Citation networks**: Build networks showing reference relationships

### Advanced Features
- **PDF caching**: Intelligent caching system for downloaded papers
- **Section extraction**: Automatically identify abstract, methodology, results, etc.
- **Rate limiting**: Respects arXiv API limits (3 requests/second)
- **Rich formatting**: Clean, readable output with proper LaTeX handling

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd arxiv-mcp-server
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Configuration

### Smart PDF Extraction Setup

The smart extractor uses three tiers for optimal PDF processing:

#### FAST Tier (Default - No Setup Required)
- Uses built-in pdfplumber + PyPDF2
- Works immediately after installation
- ~1s processing, ~70% quality

#### SMART Tier (Requires External Tools)

**Option 1: NOUGAT (Recommended for academic papers with math)**
```bash
# Install NOUGAT for neural OCR
pip install "nougat-ocr[api]>=0.1.17"

# Test installation
nougat --help
```

**Option 2: GROBID (Recommended for structured extraction)**
```bash
# Install GROBID client
pip install "grobid-client-python>=0.8.0"

# Run GROBID server (Docker recommended)
docker run --rm -it --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

#### PREMIUM Tier (API Required)

**Mistral OCR API Setup**
```bash
export MISTRAL_API_KEY="your-mistral-api-key-here"
```

### Environment Variables

- **MISTRAL_API_KEY**: Required for premium tier extraction
- **GROBID_SERVER**: GROBID server URL (default: http://localhost:8070)

### Fallback Behavior

The smart extractor automatically falls back:
- PREMIUM → SMART → FAST
- NOUGAT failure → GROBID → Enhanced basic extraction
- Missing API keys or tools automatically use available methods

**Note:** All tiers work independently. Missing dependencies only affect that specific tier.

## Usage

### Running the Server

Start the MCP server:
```bash
poetry run arxiv-mcp-server
```

### Available Tools

#### Basic Search & Discovery

#### 1. search_papers
Search for academic papers on arXiv.

**Parameters:**
- `query` (required): Search query (keywords, titles, authors)
- `max_results` (optional): Maximum number of results (default: 10, max: 100)
- `categories` (optional): Filter by arXiv categories (e.g., ["cs.AI", "math.CO"])
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)

**Example:**
```json
{
  "query": "machine learning transformers",
  "max_results": 5,
  "categories": ["cs.AI", "cs.LG"],
  "start_date": "2023-01-01"
}
```

#### 2. get_paper_details
Get detailed information about a specific paper.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID (e.g., "2301.00001" or "1234.5678v2")

**Example:**
```json
{
  "arxiv_id": "2301.00001"
}
```

#### 3. get_recent_papers
Get recent papers from specific arXiv categories.

**Parameters:**
- `category` (required): arXiv category (e.g., "cs.AI", "math.CO", "physics.gen-ph")
- `max_results` (optional): Maximum number of results (default: 10, max: 50)
- `days_back` (optional): Number of days to look back (default: 7, max: 30)

**Example:**
```json
{
  "category": "cs.AI",
  "max_results": 10,
  "days_back": 14
}
```

#### 4. get_author_papers
Find all papers by a specific author with smart name matching.

**Parameters:**
- `author_name` (required): Author name to search for
- `max_results` (optional): Maximum number of results (default: 20, max: 100)
- `categories` (optional): Filter by arXiv categories
- `start_date` (optional): Start date filter (YYYY-MM-DD)
- `end_date` (optional): End date filter (YYYY-MM-DD)

#### 5. find_similar_papers
Find papers similar to a reference paper.

**Parameters:**
- `reference_paper_id` (required): arXiv ID of the reference paper
- `max_results` (optional): Maximum number of similar papers (default: 10, max: 20)
- `similarity_method` (optional): Method for similarity (keywords, categories, authors)

#### Paper Content Analysis

#### 6. download_and_read_paper
Download and extract full text content from an ArXiv paper.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to download and read
- `format_type` (optional): Format to download (pdf or tex, default: pdf)
- `force_download` (optional): Force re-download even if cached

#### 7. summarize_paper
Generate a structured summary of a paper with key sections.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to summarize

#### 8. extract_key_findings
Extract key findings, contributions, and methodology from a paper.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to analyze

#### 9. compare_papers
Compare multiple papers across different aspects.

**Parameters:**
- `paper_ids` (required): List of arXiv paper IDs to compare (2-5 papers)
- `comparison_aspects` (optional): Aspects to compare (methodology, results, contributions, scope)

#### Citation & Bibliography Tools

#### 10. format_citation
Format a paper citation in various academic styles.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to format
- `style` (optional): Citation style (apa, mla, chicago, bibtex, default: apa)

#### 11. export_bibliography
Export multiple papers as a formatted bibliography.

**Parameters:**
- `arxiv_ids` (required): List of arXiv paper IDs to include
- `style` (optional): Citation style (apa, mla, chicago, bibtex, default: apa)

#### 12. find_citing_papers
Find papers that cite a given ArXiv paper using Semantic Scholar.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to find citations for
- `max_results` (optional): Maximum number of citing papers (default: 20, max: 50)

#### 13. get_citation_network
Build a citation network around a paper showing references and citations.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to build network around
- `depth` (optional): Network depth (default: 2, max: 3)

#### Advanced PDF Extraction

#### 14. smart_extract_paper
Advanced PDF extraction with three-tier adaptive mechanism for optimal quality.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to extract content from
- `extraction_tier` (optional): Force specific tier (fast, smart, premium)
- `budget_mode` (optional): If true, avoid paid services (default: false)
- `force_analysis` (optional): Always analyze difficulty even with user-specified tier (default: false)

**Extraction Tiers:**
- **FAST**: pdfplumber + PyPDF2 (simple papers, ~1s, ~70% quality)
- **SMART**: NOUGAT → GROBID → Enhanced fallback (moderate complexity, ~5-10s, ~85-90% quality)
- **PREMIUM**: Mistral OCR v2 (math/complex layouts, ~10s, ~95% quality)

**Example:**
```json
{
  "arxiv_id": "2301.00001",
  "extraction_tier": "smart",
  "budget_mode": false
}
```

#### 15. analyze_paper_difficulty
Analyze PDF complexity to determine optimal extraction method without extraction.

**Parameters:**
- `arxiv_id` (required): arXiv paper ID to analyze

**Analysis Factors:**
- Math density (equations, symbols, Greek letters)
- Layout complexity (multi-column, figures, tables)
- Text extractability (OCR artifacts, scanned content)
- File size and page count

**Example:**
```json
{
  "arxiv_id": "2301.00001"
}
```

### Integration with Claude Code

1. Add the server to your Claude Code configuration file (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "poetry",
      "args": ["run", "arxiv-mcp-server"],
      "cwd": "/path/to/arxiv-mcp-server"
    }
  }
}
```

2. Restart Claude Code to load the server.

3. You can now use comprehensive arXiv tools directly in Claude Code:
   - "Search for recent papers on quantum computing"
   - "Read and summarize this paper: 2301.00001"
   - "Use smart extraction on paper 2301.00001 with premium quality"
   - "Analyze the difficulty of extracting content from paper 2301.00001"
   - "Find papers similar to arXiv:2301.00001"
   - "Compare these three papers on machine learning"
   - "Format a citation for paper 2301.00001 in APA style"
   - "Find papers that cite this work"
   - "Get all papers by author 'Geoffrey Hinton'"

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black .
poetry run ruff check .
```

### Project Structure

```
arxiv-mcp-server/
   src/arxiv_mcp_server/     # Main package
      __init__.py
      server.py             # MCP server implementation
      arxiv_client.py       # arXiv API client
   mains/                    # Executable scripts
      server.py             # Main entry point
   tests/                    # Test files
      test_server.py
      test_arxiv_client.py
   logs/                     # Log files
   pyproject.toml            # Project configuration
   README.md
```

## arXiv Categories

Common arXiv categories you can use:

- **Computer Science**: `cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`, `cs.CR`, `cs.DS`
- **Mathematics**: `math.CO`, `math.NT`, `math.AG`, `math.GT`
- **Physics**: `physics.gen-ph`, `astro-ph`, `cond-mat`, `hep-ph`
- **Quantitative Biology**: `q-bio.BM`, `q-bio.CB`, `q-bio.GN`
- **Economics**: `econ.EM`, `econ.GN`, `econ.TH`

For a complete list, see: https://arxiv.org/category_taxonomy

## Rate Limiting

This server respects arXiv's API guidelines:
- Maximum 3 requests per second
- Built-in rate limiting with async queue management
- Automatic retry and backoff for API errors

## Error Handling

The server includes comprehensive error handling:
- Invalid arXiv IDs
- Network timeouts and API errors
- Malformed XML responses
- Date parsing errors
- Rate limit exceeded

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- arXiv for providing the free API
- The MCP team for the excellent protocol and SDK
- The Claude Code team for MCP integration