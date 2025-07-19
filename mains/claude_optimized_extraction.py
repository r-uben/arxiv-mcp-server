#!/usr/bin/env python3
"""Create Claude-optimized extraction format for academic papers."""

import asyncio
import sys
from pathlib import Path

# Add src to path to import directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arxiv_mcp_server.extraction.smart_extractor import (
    SmartPDFExtractor, 
    ExtractionTier,
    check_grobid_available
)

def format_for_claude(extraction_result):
    """Format extraction result optimally for Claude consumption."""
    extraction = extraction_result.get("extraction", {})
    metadata = extraction.get("metadata", {})
    sections = extraction.get("sections", {})
    references = extraction.get("references", [])
    content = extraction.get("content", "")
    
    # Build Claude-optimized format
    output = []
    
    # Header with metadata
    output.append("# ACADEMIC PAPER ANALYSIS")
    output.append("=" * 60)
    output.append("")
    
    if metadata.get("title"):
        output.append(f"**Title:** {metadata['title']}")
        output.append("")
    
    if metadata.get("authors"):
        authors = metadata["authors"]
        if len(authors) <= 5:
            output.append(f"**Authors:** {', '.join(authors)}")
        else:
            output.append(f"**Authors:** {', '.join(authors[:5])} (+{len(authors)-5} more)")
        output.append("")
    
    # Extraction quality info
    method = extraction.get("extraction_method", "unknown")
    quality = extraction.get("quality_estimate", 0)
    output.append(f"**Extraction Method:** {method} (Quality: {quality:.0%})")
    output.append(f"**Content Length:** {len(content):,} characters")
    output.append("")
    
    # Structured sections (if available from GROBID)
    if sections:
        output.append("## DOCUMENT STRUCTURE")
        output.append("")
        
        # Priority order for sections
        section_order = [
            "abstract", "introduction", "background", "related work", 
            "methodology", "methods", "approach", "design", "implementation",
            "results", "evaluation", "discussion", "conclusion", "future work"
        ]
        
        # Add sections in logical order
        processed_sections = set()
        
        for priority_section in section_order:
            for section_name, section_content in sections.items():
                if priority_section in section_name.lower() and section_name not in processed_sections:
                    if section_content.strip():
                        output.append(f"### {section_name.upper()}")
                        output.append("")
                        output.append(section_content.strip())
                        output.append("")
                        processed_sections.add(section_name)
                        break
        
        # Add remaining sections
        for section_name, section_content in sections.items():
            if section_name not in processed_sections and section_content.strip():
                output.append(f"### {section_name.upper()}")
                output.append("")
                output.append(section_content.strip())
                output.append("")
    
    else:
        # Fallback to full content if no structured sections
        output.append("## FULL PAPER CONTENT")
        output.append("")
        output.append(content)
        output.append("")
    
    # References section
    if references:
        output.append("## REFERENCES")
        output.append("")
        output.append(f"*{len(references)} references found*")
        output.append("")
        
        # Show first few references as examples
        for i, ref in enumerate(references[:5]):
            output.append(f"{i+1}. {ref}")
        
        if len(references) > 5:
            output.append(f"... and {len(references) - 5} more references")
        output.append("")
    
    # Technical details footer
    output.append("---")
    output.append("## EXTRACTION METADATA")
    output.append(f"- Processing time: {extraction.get('processing_time', 'unknown')}")
    output.append(f"- Extraction tier: {extraction.get('tier_used', method)}")
    if extraction.get("fallback_reason"):
        output.append(f"- Fallback reason: {extraction['fallback_reason']}")
    
    return "\n".join(output)

async def extract_for_claude(pdf_path, tier_preference=None):
    """Extract paper in Claude-optimized format."""
    print(f"📄 Extracting paper for Claude: {pdf_path}")
    
    if not Path(pdf_path).exists():
        return f"❌ PDF not found: {pdf_path}"
    
    # Initialize extractor
    extractor = SmartPDFExtractor()
    
    try:
        # Extract with specified tier or auto-detect
        result = await extractor.extract_paper(
            Path(pdf_path),
            user_preference=tier_preference,
            force_analysis=True
        )
        
        if not result.get("success"):
            return f"❌ Extraction failed: {result.get('error', 'Unknown error')}"
        
        # Format for Claude
        claude_format = format_for_claude(result)
        
        # Save to file
        output_file = Path(pdf_path).stem + "_for_claude.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(claude_format)
        
        print(f"✅ Claude-optimized extraction saved to: {output_file}")
        print(f"   Content: {len(claude_format):,} characters")
        
        return claude_format
        
    except Exception as e:
        return f"❌ Extraction error: {e}"

async def main():
    """Test Claude-optimized extraction."""
    test_pdf = "test_paper.pdf"
    
    if not Path(test_pdf).exists():
        print("❌ No test_paper.pdf found")
        print("   Download: curl -L -o test_paper.pdf 'https://arxiv.org/pdf/2301.00001.pdf'")
        return
    
    print("🧠 Creating Claude-optimized extraction...")
    
    # Test with SMART tier (best quality)
    result = await extract_for_claude(test_pdf, ExtractionTier.SMART)
    
    if result.startswith("❌"):
        print(result)
    else:
        print("\n" + "="*60)
        print("📋 PREVIEW FOR CLAUDE (first 2000 chars)")
        print("="*60)
        print(result[:2000] + "..." if len(result) > 2000 else result)

if __name__ == "__main__":
    asyncio.run(main())