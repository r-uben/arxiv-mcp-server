# ArXiv MCP Server Testing Guide

## Fixed Issues ✅

The following issues have been resolved in the ArXiv MCP server:

### 1. **Tier Selection Problem** 
- **Issue**: MCP server was using FAST tier (pdfplumber) instead of SMART tier (GROBID)
- **Fix**: Added FORCE_SMART environment variable and improved health checks
- **Result**: Now uses GROBID with 85% quality vs 70% with FAST tier

### 2. **GROBID Connectivity**
- **Issue**: Health check failing in MCP context 
- **Fix**: Enhanced health check with multiple endpoints (/api/health, /api/isalive, /)
- **Result**: Proper GROBID server detection and connection

### 3. **Processing Speed**
- **Issue**: 1 second processing (immediate fallback to FAST)
- **Fix**: Proper GROBID processing with realistic timeouts
- **Result**: 30-60 seconds processing time with structured output

### 4. **Quality & Structure**
- **Issue**: Missing sections, poor formatting, concatenated text
- **Fix**: TEI XML parsing with proper section extraction
- **Result**: Structured academic paper format with proper sections

## How to Test

### Start GROBID Server
```bash
docker run --rm -d --init -p 8070:8070 --name grobid-server lfoppiano/grobid:0.8.0
```

### Start MCP Server
```bash
./run-server.sh
```

### Test in Claude Desktop
Use this prompt:
```
I want to test my ArXiv MCP server. Please extract and analyze the paper "NFTrig: Using Blockchain Technologies for Math Education" from ArXiv ID 2301.00001. 

After extraction, please provide:
1. The paper's title, authors, and abstract
2. A summary of the main sections and their key points
3. The specific blockchain technologies and programming languages used
4. The extraction method and processing time
5. The quality/confidence of the extraction

This will help me verify that my GROBID integration is working correctly.
```

### Expected Results ✅

**Processing Time**: 30-60 seconds (not 1 second)
**Extraction Method**: GROBID/TEI (not pdfplumber)
**Quality**: 85%+ (not 70%)
**Content Length**: 45,000+ characters with proper structure
**Sections**: Abstract, Introduction, Methods, Results, etc.

### Debug Commands

Check GROBID health:
```bash
curl http://localhost:8070/api/health
```

Test extraction locally:
```bash
poetry run python mains/test_mcp_fixes.py
```

Check MCP server logs for tier selection:
```bash
tail -f logs/arxiv_mcp_server.log | grep "extraction complete"
```

## Environment Variables

- `FORCE_SMART=true` - Force SMART tier for academic papers (default: enabled)
- `GROBID_SERVER=http://localhost:8070` - GROBID server URL
- `TIER_SELECTOR_DEBUG=true` - Enable debug logging for tier selection

## Troubleshooting

### If FAST tier is still used:
1. Check GROBID server is running: `docker ps | grep grobid`
2. Check health endpoint: `curl http://localhost:8070/api/health`
3. Verify FORCE_SMART is enabled in MCP server logs
4. Check for network connectivity between MCP and GROBID

### If processing is too fast (1-2 seconds):
- This indicates fallback to FAST tier
- Check GROBID connectivity
- Look for error messages in logs

### If quality is low (70% or less):
- Indicates pdfplumber was used instead of GROBID
- Verify GROBID server is responding
- Check FORCE_SMART environment variable

## Success Indicators

✅ **Processing Time**: 30-60 seconds  
✅ **Method**: grobid_tei  
✅ **Quality**: 85%+  
✅ **Content**: 45,000+ characters  
✅ **Structure**: Proper sections (Abstract, Introduction, etc.)  
✅ **Metadata**: Title, authors, references extracted  

## Architecture

```
Claude Desktop
    ↓
ArXiv MCP Server (FORCE_SMART=true)
    ↓
SmartPDFExtractor
    ↓
GROBID Server (Docker) → TEI XML → Structured Content
```

The system now properly routes academic papers through GROBID for optimal extraction quality and structure.