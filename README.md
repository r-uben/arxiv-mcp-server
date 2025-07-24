# arXiv MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io)

I built this MCP server to access 2.4M+ arXiv papers directly in Claude Desktop. It uses GROBID for academic PDF extraction and builds citation networks to track research connections.

## What It Does

- Search arXiv by keywords, authors, categories, and dates
- Extract full text from PDFs using GROBID (handles equations and references)
- Build citation networks using Semantic Scholar integration
- Manage a local library with collections and tags
- Generate summaries and compare papers side-by-side

## PDF Extraction

I implemented three extraction tiers that adapt to document complexity:

- **FAST**: pdfplumber for simple documents (~1s)
- **SMART**: GROBID for academic papers (~5s) - preserves equations and references
- **PREMIUM**: Mistral OCR for complex layouts (~2s) - requires API key

## 🚀 Quick Start

### Installation

#### Option 1: Install via npm (Recommended)

```bash
# Install globally
npm install -g arxiv-mcp-server

# Or install locally in a project
npm install arxiv-mcp-server
```

#### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/r-uben/arxiv-mcp-server.git
cd arxiv-mcp-server

# Install dependencies with Poetry
poetry install

# Test the server
poetry run arxiv-mcp-server
```

### Claude Desktop Integration

#### For npm installation:

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "npx",
      "args": ["arxiv-mcp-server"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

Or for global installation:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "arxiv-mcp-server"
    }
  }
}
```

#### For Poetry installation:

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

Restart Claude Desktop and you're ready to go!

## Examples

```
"Search for recent papers on large language models in the last 6 months"
"Find all papers by Geoffrey Hinton on deep learning"
"Build a citation network around paper 2301.00001"
"Save paper 2301.00001 to my 'Transformers' collection"
"Summarize the key findings from paper 2301.00001"
```

## ⚙️ Configuration

### API Keys (Optional)

For enhanced features, set these environment variables:

```bash
# For premium PDF extraction (Mistral OCR)
export MISTRAL_API_KEY="your-mistral-api-key"

# For faster citation lookups (Semantic Scholar)
export SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-api-key"
```

### External Services (Optional)

**GROBID Server** - For enhanced academic paper processing:

```bash
docker run --rm -it --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

### Configuration Options

| Variable | Purpose | Default |
|----------|---------|---------|
| `MISTRAL_API_KEY` | Premium OCR extraction | None |
| `SEMANTIC_SCHOLAR_API_KEY` | Citation discovery API | None |
| `GROBID_SERVER` | GROBID server URL | `http://localhost:8070` |
| `FORCE_SMART` | Always use SMART tier for academic papers | `true` |

## Available Tools

I've implemented 25 tools across four categories:

- **Search & Discovery**: search papers, find by author, get recent papers, find similar papers
- **Library Management**: save papers, manage collections, track reading status, search library
- **Citation Analysis**: extract references, find citing papers, build citation networks
- **Content Analysis**: extract PDFs, summarize papers, compare papers, extract key findings

## How It Works

The server automatically:
1. Analyzes PDF complexity and selects the best extraction method
2. Caches papers locally to reduce API calls
3. Respects rate limits (arXiv: 3 req/s, Semantic Scholar: 1-4 req/s)
4. Falls back gracefully when services are unavailable



## Development

```bash
# Development setup
poetry install
poetry run pytest                    # Run tests
poetry run black .                   # Format code  
poetry run ruff check .              # Lint code

# Testing individual components
poetry run python -m pytest tests/  # Full test suite
poetry run arxiv-mcp-server          # Start server manually
```

## arXiv Categories

| Field | Popular Categories |
|-------|-------------------|
| **Computer Science** | `cs.AI`, `cs.LG`, `cs.CV`, `cs.CL`, `cs.RO` |
| **Mathematics** | `math.CO`, `math.NT`, `math.AG`, `math.ST` |
| **Physics** | `astro-ph`, `cond-mat`, `hep-ph`, `quant-ph` |
| **Biology** | `q-bio.BM`, `q-bio.CB`, `q-bio.GN` |

[Complete arXiv taxonomy →](https://arxiv.org/category_taxonomy)


## License

MIT License © 2025 Ruben Fernández-Fuertes

