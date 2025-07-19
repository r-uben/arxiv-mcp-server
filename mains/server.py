"""Main entry point for the arXiv MCP server."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


async def async_main():
    """Async main entry point."""
    logger.info("Starting arXiv MCP Server...")

    try:
        server = create_server()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def main():
    """Main entry point for Poetry script."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
