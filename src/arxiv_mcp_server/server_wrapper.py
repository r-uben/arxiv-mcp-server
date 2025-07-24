#!/usr/bin/env python3
"""Wrapper to ensure clean MCP server startup."""

import sys
import os
import io
from contextlib import redirect_stdout

# Immediately redirect stdout to prevent any output during imports
_null_stdout = io.StringIO()
sys.stdout = _null_stdout

def main():
    """Run the MCP server with stdout protection."""
    # Import after stdout redirection
    from arxiv_mcp_server.server import main as server_main
    
    # Restore stdout for MCP protocol
    sys.stdout = sys.__stdout__
    
    # Run the server
    server_main()

if __name__ == "__main__":
    main()