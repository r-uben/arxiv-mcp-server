"""
Docker-based Nougat extractor to avoid dependency conflicts.
"""

import asyncio
import tempfile
import json
import aiohttp
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class NougatDockerExtractor:
    """Nougat extractor using Docker container with HTTP API."""
    
    def __init__(self, api_url: str = "http://localhost:8503"):
        self.api_url = api_url
        self.container_name = "nougat-server"
        
    async def is_container_running(self) -> bool:
        """Check if Nougat Docker container is running."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            running_containers = stdout.decode().strip().split('\n')
            return self.container_name in running_containers
            
        except Exception as e:
            logger.debug(f"Error checking container status: {e}")
            return False
    
    async def start_container(self) -> bool:
        """Start Nougat Docker container."""
        try:
            logger.info("Starting Nougat Docker container...")
            
            # Check if container exists but is stopped
            process = await asyncio.create_subprocess_exec(
                "docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            
            if self.container_name in stdout.decode():
                # Container exists, start it
                process = await asyncio.create_subprocess_exec(
                    "docker", "start", self.container_name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            else:
                # Container doesn't exist, create and run it
                process = await asyncio.create_subprocess_exec(
                    "docker", "run", "-d", "--name", self.container_name,
                    "-p", "8503:8503",
                    "nougat:latest",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
            
            # Wait for container to be ready
            await asyncio.sleep(5)
            
            # Check if it's running
            return await self.is_container_running()
            
        except Exception as e:
            logger.error(f"Failed to start Nougat container: {e}")
            return False
    
    async def extract_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract document using Nougat Docker container."""
        
        # Check if container is running
        if not await self.is_container_running():
            logger.info("Nougat container not running, attempting to start...")
            if not await self.start_container():
                raise Exception("Failed to start Nougat Docker container")
        
        try:
            # Prepare file for upload
            async with aiohttp.ClientSession() as session:
                with open(pdf_path, 'rb') as file:
                    data = aiohttp.FormData()
                    data.add_field('file', file, filename=pdf_path.name, content_type='application/pdf')
                    
                    # Send request to Nougat API
                    async with session.post(
                        f"{self.api_url}/predict/",
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.text()
                            
                            return {
                                "content": result,
                                "sections": self._parse_sections(result),
                                "extraction_method": "nougat_docker",
                                "processing_time": "~15s",
                                "quality_estimate": 0.90,
                                "format": "mathpix_markdown"
                            }
                        else:
                            error_text = await response.text()
                            raise Exception(f"Nougat API error {response.status}: {error_text}")
                            
        except Exception as e:
            logger.error(f"Nougat Docker extraction failed: {e}")
            raise
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse sections from Nougat markdown output."""
        sections = {}
        current_section = "abstract"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Detect section headers (# ## ###)
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.strip('#').strip().lower().replace(' ', '_')
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections


async def check_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        return process.returncode == 0
    except Exception:
        return False


async def check_nougat_docker_image() -> bool:
    """Check if Nougat Docker image is available."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "images", "--filter", "reference=nougat", "--format", "{{.Repository}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        return "nougat" in stdout.decode()
        
    except Exception:
        return False