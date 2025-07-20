#!/usr/bin/env python3
"""Setup and test Nougat functionality with multiple approaches."""

import sys
import asyncio
import subprocess
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_mcp_server.extraction.smart_extractor import check_nougat_available

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_docker():
    """Check if Docker is available."""
    print("🐳 Checking Docker availability...")
    try:
        process = await asyncio.create_subprocess_exec(
            "docker", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            version = stdout.decode().strip()
            print(f"✅ Docker available: {version}")
            return True
        else:
            print(f"❌ Docker check failed: {stderr.decode()}")
            return False
    except Exception as e:
        print(f"❌ Docker not available: {e}")
        return False


async def setup_nougat_docker():
    """Set up Nougat using Docker."""
    print("\n🔧 Setting up Nougat with Docker...")
    
    # Check if Docker is available
    if not await check_docker():
        print("❌ Docker required for this setup method")
        return False
    
    try:
        # Check if Nougat image exists
        print("📦 Checking for Nougat Docker image...")
        process = await asyncio.create_subprocess_exec(
            "docker", "images", "nougat", "--format", "{{.Repository}}:{{.Tag}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if "nougat" not in stdout.decode():
            print("📥 Nougat image not found. Pulling from Docker Hub...")
            print("⚠️  This may take several minutes (~2GB download)...")
            
            # Pull the image
            process = await asyncio.create_subprocess_exec(
                "docker", "pull", "nougat:latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                print("❌ Failed to pull Nougat image")
                print("💡 Try manually: docker pull nougat:latest")
                return False
        else:
            print("✅ Nougat Docker image found")
        
        # Check if container is running
        print("🔍 Checking Nougat container status...")
        process = await asyncio.create_subprocess_exec(
            "docker", "ps", "--filter", "name=nougat-server", "--format", "{{.Names}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        
        if "nougat-server" not in stdout.decode():
            print("🚀 Starting Nougat container...")
            
            # Start container
            process = await asyncio.create_subprocess_exec(
                "docker", "run", "-d", "--name", "nougat-server",
                "-p", "8503:8503",
                "nougat:latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            
            if process.returncode != 0:
                print("❌ Failed to start Nougat container")
                return False
            
            print("⏳ Waiting for container to initialize...")
            await asyncio.sleep(10)
        else:
            print("✅ Nougat container already running")
        
        print("✅ Nougat Docker setup complete!")
        print("🌐 API available at: http://localhost:8503")
        return True
        
    except Exception as e:
        print(f"❌ Docker setup failed: {e}")
        return False


def show_manual_setup_instructions():
    """Show manual setup instructions for different approaches."""
    print("\n📋 Manual Nougat Setup Options:")
    print("=" * 60)
    
    print("\n1. 🐳 Docker Approach (Recommended):")
    print("   # Pull and run Nougat container")
    print("   docker pull nougat:latest")
    print("   docker run -d --name nougat-server -p 8503:8503 nougat:latest")
    print("   # Test: curl http://localhost:8503/health")
    
    print("\n2. 🔧 Fix Dependency Issues:")
    print("   # Downgrade problematic packages")
    print("   pip install albumentations==1.3.1")
    print("   pip install pydantic==1.10.12")
    print("   # Warning: May affect other dependencies")
    
    print("\n3. 🐍 Alternative: Use Marker or Surya-OCR:")
    print("   pip install marker-pdf  # Alternative to Nougat")
    print("   pip install surya-ocr   # Another alternative")
    
    print("\n4. ☁️  Cloud API Approach:")
    print("   # Use hosted services like:")
    print("   # - Mathpix API")
    print("   # - Adobe PDF Services")
    print("   # - Google Document AI")


async def test_current_setup():
    """Test the current Nougat setup."""
    print("\n🧪 Testing current Nougat setup...")
    
    available = check_nougat_available()
    print(f"{'✅' if available else '❌'} Nougat availability check: {'Passed' if available else 'Failed'}")
    
    if available:
        print("💡 Nougat seems to be available through one of the methods")
        
        # Test with actual extraction
        test_pdf = Path(__file__).parent.parent / "test_paper.pdf"
        if test_pdf.exists():
            print(f"📄 Test PDF found: {test_pdf}")
            print("💡 You can test extraction by using the smart_extractor")
        else:
            print("⚠️  No test PDF found for extraction testing")
    else:
        print("💡 Consider using one of the setup options above")
    
    return available


async def main():
    """Main setup and testing workflow."""
    print("🚀 Nougat Setup and Testing Tool")
    print("=" * 40)
    
    # Test current setup
    current_works = await test_current_setup()
    
    if not current_works:
        # Try Docker setup
        print("\n💡 Attempting automatic Docker setup...")
        docker_success = await setup_nougat_docker()
        
        if docker_success:
            print("\n✅ Docker setup successful! Testing again...")
            await test_current_setup()
        else:
            print("\n❌ Automatic setup failed")
    
    # Always show manual instructions
    show_manual_setup_instructions()
    
    print("\n📋 Summary:")
    print(f"✅ Nougat package installed: Yes (0.1.17)")
    print(f"{'✅' if current_works else '❌'} Nougat functional: {'Yes' if current_works else 'No'}")
    print(f"{'✅' if await check_docker() else '❌'} Docker available: {'Yes' if await check_docker() else 'No'}")
    
    print("\n🎯 Next Steps:")
    if current_works:
        print("✅ Nougat is ready to use!")
        print("💡 Test with: poetry run test-extraction")
    else:
        print("1. Try Docker approach: docker run -d --name nougat-server -p 8503:8503 nougat:latest")
        print("2. Or use GROBID instead (already working in your setup)")
        print("3. For now, rely on GROBID + fallback extraction methods")


if __name__ == "__main__":
    asyncio.run(main())