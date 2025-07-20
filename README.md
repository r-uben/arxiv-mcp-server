# arXiv MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Model Context Protocol (MCP) server that provides intelligent access to arXiv's academic paper repository. This server transforms how you interact with scientific literature by offering advanced search, content analysis, and citation management capabilities through any MCP-compatible interface like Claude Code.

## ✨ Key Features

### 🔍 **Smart Search & Discovery**

- **Advanced Search**: Multi-faceted search by keywords, authors, categories, and date ranges
- **Author Intelligence**: Smart name matching and comprehensive author paper discovery
- **Category Filtering**: Browse recent papers by specific arXiv subject areas
- **Similarity Detection**: Find related papers through intelligent content analysis

### 📄 **Intelligent Content Analysis**

- **Adaptive PDF Processing**: Three-tier extraction system (FAST/SMART/PREMIUM) automatically selects optimal method
- **Full-Text Extraction**: Complete paper content with mathematical formulations preserved
- **Structured Summarization**: AI-powered summaries highlighting key contributions and methodology
- **Comparative Analysis**: Side-by-side comparison of multiple papers across different dimensions

### 📚 **Professional Citation Management**

- **Multi-Format Citations**: Generate citations in APA, MLA, Chicago, and BibTeX formats
- **Bibliography Export**: Create publication-ready reference lists
- **Citation Networks**: Discover citation relationships and academic influence
- **Reference Tracking**: Find papers that cite specific works (via Semantic Scholar integration)

### ⚡ **Performance & Reliability**

- **Intelligent Caching**: Efficient PDF storage and retrieval system
- **Rate Limit Compliance**: Respects arXiv API guidelines (3 requests/second)
- **Graceful Fallbacks**: Automatic tier downgrade when premium services unavailable
- **Rich Text Processing**: Handles LaTeX formatting and mathematical notation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Poetry package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd arxiv-mcp-server

# Install dependencies
poetry install

# Start the server
poetry run arxiv-mcp-server
```

## ⚙️ Configuration

### PDF Extraction Tiers

The server automatically selects the best extraction method based on document complexity and available tools:

| Tier | Speed | Quality | Requirements | Best For |
|------|-------|---------|-------------|----------|
| **FAST** | ~1s | ~70% | Built-in (default) | Simple text documents |
| **SMART** | ~5-10s | ~85-90% | External tools | Academic papers with math |
| **PREMIUM** | ~10s | ~95% | API key | Complex layouts, heavy math |

### Optional Enhancements

#### For SMART Tier (Choose one or both)

**NOUGAT** - Neural OCR for mathematical content:

```bash
pip install "nougat-ocr[api]>=0.1.17"
```

**GROBID** - Structured document parsing:

```bash
# Install client
pip install "grobid-client-python>=0.8.0"

# Run server (Docker)
docker run --rm -it --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

#### For PREMIUM Tier

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `MISTRAL_API_KEY` | Premium OCR extraction | None |
| `GROBID_SERVER` | GROBID server URL | `http://localhost:8070` |

> **Note**: Missing dependencies only affect their specific tier. The system gracefully falls back to available methods.

## 🔧 MCP Integration

### Claude Code Setup

Add to your `~/.claude/claude_desktop_config.json`:

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

Restart Claude Code and start using commands like:

- *"Search for recent papers on quantum computing"*
- *"Read and summarize paper 2301.00001"*
- *"Find papers similar to arXiv:2301.00001"*
- *"Format a citation for paper 2301.00001 in APA style"*

## 📖 API Reference

### Available Tools

<details>

<summary><strong>📋 Search & Discovery</strong></summary>

**Search Tools:**

- `search_papers` - Advanced search with keyword, author, category, and date filters
- `get_recent_papers` - Latest papers from specific arXiv categories
- `get_author_papers` - Find all papers by specific authors with smart matching
- `find_similar_papers` - Discover related papers using various similarity methods
- `get_paper_details` - Detailed metadata for specific paper IDs

</details>

<details>

<summary><strong>📄 Content Analysis</strong></summary>

**Extraction Tools:**

- `smart_extract_paper` - Advanced PDF extraction with three-tier quality system
- `download_and_read_paper` - Full text extraction with format options
- `analyze_paper_difficulty` - Assess PDF complexity for optimal extraction method

**Analysis Tools:**

- `summarize_paper` - Generate structured summaries with key insights
- `extract_key_findings` - Identify contributions, methodology, and results
- `compare_papers` - Multi-paper comparative analysis across dimensions

</details>

<details>

<summary><strong>📚 Citation Management</strong></summary>

**Citation Tools:**

- `format_citation` - Generate citations in APA, MLA, Chicago, BibTeX formats
- `export_bibliography` - Create publication-ready reference lists
- `find_citing_papers` - Discover papers that reference a work (via Semantic Scholar)
- `get_citation_network` - Build citation relationship networks

</details>

### Quick Reference

**Example Usage:**

```json
{
  "query": "machine learning transformers",
  "max_results": 5,
  "categories": ["cs.AI", "cs.LG"]
}
```

For detailed parameter documentation, see the [MCP protocol specification](https://github.com/modelcontextprotocol/spec).

## 🛠️ Development

```bash
# Quick commands
poetry run pytest          # Run tests
poetry run black .         # Format code
poetry run ruff check .    # Lint code
poetry run arxiv-mcp-server # Start server
```

## 📚 arXiv Categories Reference

| Field | Examples |
|-------|----------|
| **Computer Science** | `cs.AI`, `cs.LG`, `cs.CV`, `cs.CL` |
| **Mathematics** | `math.CO`, `math.NT`, `math.AG` |
| **Physics** | `astro-ph`, `cond-mat`, `hep-ph` |
| **Biology** | `q-bio.BM`, `q-bio.CB`, `q-bio.GN` |

[Complete taxonomy →](https://arxiv.org/category_taxonomy)

## ⚡ Performance Notes

- **Rate Limiting**: Complies with arXiv's 3 requests/second limit
- **Smart Caching**: Reduces redundant downloads and API calls
- **Graceful Degradation**: Automatic fallbacks for unavailable services
- **Error Recovery**: Comprehensive handling for network issues and malformed data

## 🤝 Contributing

Contributions welcome! Please fork the repository, create a feature branch, add tests, and submit a pull request.

```bash
# Development workflow
git checkout -b feature/your-feature
poetry install
poetry run pytest
poetry run black .
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built with ❤️ for the research community**

*Special thanks to arXiv for their open API, the MCP team for the excellent protocol, and Claude Code for seamless integration.*