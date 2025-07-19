"""
arXiv MCP Server - Simple entry point matching zen-mcp-server pattern
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from arxiv_mcp_server.server import create_server

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/arxiv_mcp_server.log"),
        logging.StreamHandler(sys.stderr),
    ],
)

logger = logging.getLogger(__name__)


async def main():
    """Main async entry point."""
    logger.info("Starting arXiv MCP Server...")
    
    try:
        server = create_server()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def run():
    """Console script entry point for arxiv-mcp-server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handle graceful shutdown
        pass


if __name__ == "__main__":
    run()