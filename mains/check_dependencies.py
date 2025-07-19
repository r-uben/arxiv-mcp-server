#!/usr/bin/env python3
"""Check which extraction tools are available."""

import os
import subprocess
import requests
from pathlib import Path

def check_nougat():
    """Check if NOUGAT is installed and working."""
    try:
        result = subprocess.run(["nougat", "--help"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def check_grobid():
    """Check if GROBID server is running."""
    try:
        response = requests.get("http://localhost:8070/api/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def check_mistral_key():
    """Check if Mistral API key is configured."""
    return os.getenv("MISTRAL_API_KEY") is not None

def main():
    print("🔧 Checking PDF extraction dependencies:")
    print()
    
    # Check each tool
    nougat_ok = check_nougat()
    grobid_ok = check_grobid()
    mistral_ok = check_mistral_key()
    
    print(f"📦 NOUGAT: {'✅ Available' if nougat_ok else '❌ Not installed'}")
    if not nougat_ok:
        print("   Install: pip install 'nougat-ocr[api]>=0.1.17'")
    
    print(f"🐳 GROBID: {'✅ Running' if grobid_ok else '❌ Not running'}")
    if not grobid_ok:
        print("   Start: docker run --rm -it --init -p 8070:8070 lfoppiano/grobid:0.8.0")
    
    print(f"🔑 Mistral API: {'✅ Configured' if mistral_ok else '❌ No API key'}")
    if not mistral_ok:
        print("   Set: export MISTRAL_API_KEY='your-api-key'")
    
    print()
    print("📊 Available tiers:")
    print(f"  FAST (pdfplumber): ✅ Always available")
    print(f"  SMART (NOUGAT/GROBID): {'✅' if nougat_ok or grobid_ok else '❌'} {'Available' if nougat_ok or grobid_ok else 'Not available'}")
    print(f"  PREMIUM (Mistral OCR): {'✅' if mistral_ok else '❌'} {'Available' if mistral_ok else 'No API key'}")

if __name__ == "__main__":
    main()