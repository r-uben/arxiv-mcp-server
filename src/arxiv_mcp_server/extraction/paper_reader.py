"""Paper reading and content extraction utilities."""

import os
import re
import logging
import tempfile
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urljoin, urlparse

import pdfplumber
import PyPDF2
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PaperReader:
    """Download and extract content from ArXiv papers."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize paper reader with optional cache directory."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "arxiv_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
    async def download_and_read_paper(
        self, 
        arxiv_id: str, 
        format_type: str = "pdf",
        force_download: bool = False
    ) -> Dict[str, Any]:
        """Download and extract content from an ArXiv paper."""
        
        # Normalize ArXiv ID
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        
        # Check cache first
        cache_path = self.cache_dir / f"{arxiv_id}.{format_type}"
        
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached paper: {cache_path}")
            content = await self._extract_content_from_file(cache_path, format_type)
        else:
            # Download paper
            file_path = await self._download_paper(arxiv_id, format_type)
            content = await self._extract_content_from_file(file_path, format_type)
            
        return {
            "arxiv_id": arxiv_id,
            "format": format_type,
            "content": content,
            "cached_path": str(cache_path) if cache_path.exists() else None
        }
    
    async def _download_paper(self, arxiv_id: str, format_type: str) -> Path:
        """Download paper from ArXiv."""
        if format_type == "pdf":
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif format_type == "tex":
            url = f"https://arxiv.org/e-print/{arxiv_id}"
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        cache_path = self.cache_dir / f"{arxiv_id}.{format_type}"
        
        logger.info(f"Downloading paper from: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(cache_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    logger.info(f"Downloaded paper to: {cache_path}")
                else:
                    raise Exception(f"Failed to download paper: HTTP {response.status}")
                    
        return cache_path
    
    async def _extract_content_from_file(self, file_path: Path, format_type: str) -> Dict[str, Any]:
        """Extract content from downloaded paper file."""
        if format_type == "pdf":
            return await self._extract_pdf_content(file_path)
        elif format_type == "tex":
            return await self._extract_tex_content(file_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    async def _extract_pdf_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from PDF using multiple methods."""
        content = {
            "raw_text": "",
            "structured_content": {},
            "metadata": {},
            "page_count": 0,
            "extraction_method": "hybrid"
        }
        
        try:
            # Method 1: Try pdfplumber (better for structured content)
            with pdfplumber.open(pdf_path) as pdf:
                content["page_count"] = len(pdf.pages)
                content["metadata"] = pdf.metadata or {}
                
                text_parts = []
                structured_parts = {}
                
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {i+1} ---\n{page_text}")
                        
                        # Try to identify sections on this page
                        sections = self._identify_sections(page_text)
                        if sections:
                            structured_parts[f"page_{i+1}"] = sections
                
                content["raw_text"] = "\n\n".join(text_parts)
                content["structured_content"] = structured_parts
                
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}, trying PyPDF2")
            
            # Fallback: PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    content["page_count"] = len(pdf_reader.pages)
                    content["metadata"] = pdf_reader.metadata or {}
                    
                    text_parts = []
                    for i, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(f"--- Page {i+1} ---\n{page_text}")
                    
                    content["raw_text"] = "\n\n".join(text_parts)
                    content["extraction_method"] = "pypdf2_fallback"
                    
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                content["error"] = str(e2)
        
        # Post-process the text
        if content["raw_text"]:
            content["clean_text"] = self._clean_extracted_text(content["raw_text"])
            content["sections"] = self._extract_paper_sections(content["clean_text"])
            content["word_count"] = len(content["clean_text"].split())
        
        return content
    
    async def _extract_tex_content(self, tex_path: Path) -> Dict[str, Any]:
        """Extract content from TeX source files."""
        # This would handle .tar.gz files with LaTeX source
        # For now, return basic structure
        return {
            "raw_text": f"TeX source file: {tex_path}",
            "format": "tex",
            "note": "TeX parsing not yet implemented"
        }
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """Normalize ArXiv ID format."""
        # Remove version if present (2301.00001v2 -> 2301.00001)
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        return arxiv_id
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove multiple whitespaces and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)  # Fix sentence breaks
        
        return text.strip()
    
    def _identify_sections(self, page_text: str) -> Dict[str, str]:
        """Identify sections on a page."""
        sections = {}
        
        # Common section headers in academic papers
        section_patterns = [
            r'\b(Abstract|ABSTRACT)\b',
            r'\b(\d+\.?\s*Introduction|INTRODUCTION)\b',
            r'\b(\d+\.?\s*Related Work|RELATED WORK)\b',
            r'\b(\d+\.?\s*Methodology|METHODOLOGY|Methods|METHODS)\b',
            r'\b(\d+\.?\s*Results|RESULTS)\b',
            r'\b(\d+\.?\s*Discussion|DISCUSSION)\b',
            r'\b(\d+\.?\s*Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\b',
            r'\b(\d+\.?\s*References|REFERENCES)\b',
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                section_name = match.group(1).lower().strip()
                # Get some context around the match
                start = max(0, match.start() - 50)
                end = min(len(page_text), match.end() + 200)
                sections[section_name] = page_text[start:end].strip()
        
        return sections
    
    def _extract_paper_sections(self, text: str) -> Dict[str, str]:
        """Extract major sections from the full paper text."""
        sections = {}
        
        # More comprehensive section extraction
        patterns = {
            "abstract": r'\b(Abstract|ABSTRACT)\b[\s\n]*(.{100,2000}?)(?=\b(?:Introduction|Keywords|1\.|I\.|\n\n))',
            "introduction": r'\b(\d+\.?\s*Introduction|INTRODUCTION|1\.?\s*Introduction)\b[\s\n]*(.{200,3000}?)(?=\b(?:\d+\.|\n\n[A-Z]))',
            "methodology": r'\b(\d+\.?\s*(?:Methodology|Methods|Method|Approach))\b[\s\n]*(.{200,3000}?)(?=\b(?:\d+\.|\n\n[A-Z]))',
            "results": r'\b(\d+\.?\s*(?:Results|Experiments|Evaluation))\b[\s\n]*(.{200,3000}?)(?=\b(?:\d+\.|\n\n[A-Z]))',
            "conclusion": r'\b(\d+\.?\s*(?:Conclusion|Conclusions|Summary))\b[\s\n]*(.{100,2000}?)(?=\b(?:References|Acknowledgments|\Z))',
        }
        
        for section_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(2).strip()
        
        return sections


class PaperAnalyzer:
    """Analyze paper content and extract insights."""
    
    def __init__(self):
        self.reader = PaperReader()
    
    async def summarize_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """Generate a structured summary of the paper."""
        paper_content = await self.reader.download_and_read_paper(arxiv_id)
        
        if "error" in paper_content["content"]:
            return {"error": paper_content["content"]["error"]}
        
        content = paper_content["content"]
        sections = content.get("sections", {})
        
        summary = {
            "arxiv_id": arxiv_id,
            "metadata": {
                "page_count": content.get("page_count", 0),
                "word_count": content.get("word_count", 0),
                "extraction_method": content.get("extraction_method", "unknown")
            },
            "sections_found": list(sections.keys()),
            "abstract": sections.get("abstract", "Not found"),
            "key_sections": {},
            "overview": self._generate_overview(sections)
        }
        
        # Extract key information from each section
        for section_name, section_text in sections.items():
            if section_name in ["introduction", "methodology", "results", "conclusion"]:
                summary["key_sections"][section_name] = {
                    "length": len(section_text.split()),
                    "preview": section_text[:300] + "..." if len(section_text) > 300 else section_text
                }
        
        return summary
    
    async def extract_key_findings(self, arxiv_id: str) -> Dict[str, Any]:
        """Extract key findings and contributions from the paper."""
        paper_content = await self.reader.download_and_read_paper(arxiv_id)
        
        if "error" in paper_content["content"]:
            return {"error": paper_content["content"]["error"]}
        
        content = paper_content["content"]
        text = content.get("clean_text", "")
        sections = content.get("sections", {})
        
        findings = {
            "arxiv_id": arxiv_id,
            "contributions": self._extract_contributions(text),
            "results": self._extract_results(sections.get("results", "")),
            "methodology": self._extract_methodology(sections.get("methodology", "")),
            "conclusions": self._extract_conclusions(sections.get("conclusion", "")),
            "equations": self._extract_equations(text),
            "figures_mentioned": self._extract_figure_references(text)
        }
        
        return findings
    
    def _generate_overview(self, sections: Dict[str, str]) -> str:
        """Generate a brief overview of the paper."""
        overview_parts = []
        
        if "abstract" in sections:
            overview_parts.append(f"Abstract: {sections['abstract'][:200]}...")
        
        if "introduction" in sections:
            overview_parts.append(f"Focus: {sections['introduction'][:150]}...")
            
        if "methodology" in sections:
            overview_parts.append(f"Approach: {sections['methodology'][:150]}...")
            
        return " | ".join(overview_parts) if overview_parts else "Overview not available"
    
    def _extract_contributions(self, text: str) -> List[str]:
        """Extract main contributions from the paper."""
        contributions = []
        
        # Look for contribution keywords
        contribution_patterns = [
            r'(?:contributions?|contribute)[^.]*?(?:are|is|include)[^.]*?[.!]',
            r'(?:we|this paper|this work)\s+(?:propose|present|introduce|develop)[^.]*?[.!]',
            r'(?:main|key|primary)\s+contribution[^.]*?[.!]',
            r'(?:novel|new)\s+(?:approach|method|technique|algorithm)[^.]*?[.!]'
        ]
        
        for pattern in contribution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contributions.extend([match.strip() for match in matches])
        
        return contributions[:5]  # Return top 5
    
    def _extract_results(self, results_text: str) -> List[str]:
        """Extract key results from the results section."""
        if not results_text:
            return []
        
        results = []
        
        # Look for result patterns
        result_patterns = [
            r'(?:achieve|obtained?|reached?|showed?)[^.]*?(?:\d+\.?\d*%|improvement|better|significant)[^.]*?[.!]',
            r'(?:outperform|superior|better than)[^.]*?[.!]',
            r'(?:accuracy|precision|recall|f1)[^.]*?(?:\d+\.?\d*)[^.]*?[.!]'
        ]
        
        for pattern in result_patterns:
            matches = re.findall(pattern, results_text, re.IGNORECASE)
            results.extend([match.strip() for match in matches])
        
        return results[:3]  # Return top 3
    
    def _extract_methodology(self, methodology_text: str) -> Dict[str, str]:
        """Extract methodology information."""
        if not methodology_text:
            return {}
        
        return {
            "approach": methodology_text[:300] + "..." if len(methodology_text) > 300 else methodology_text,
            "length": len(methodology_text.split()),
            "algorithms_mentioned": len(re.findall(r'\balgorithm\b', methodology_text, re.IGNORECASE))
        }
    
    def _extract_conclusions(self, conclusion_text: str) -> List[str]:
        """Extract main conclusions."""
        if not conclusion_text:
            return []
        
        # Simple extraction - could be enhanced with NLP
        sentences = re.split(r'[.!?]+', conclusion_text)
        conclusions = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        return conclusions[:3]  # Return top 3
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        # Look for LaTeX-style equations
        equation_patterns = [
            r'\$\$(.+?)\$\$',  # Display equations
            r'\$(.+?)\$',      # Inline equations
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\begin\{align\}(.+?)\\end\{align\}'
        ]
        
        equations = []
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend([match.strip() for match in matches])
        
        return equations[:10]  # Return top 10
    
    def _extract_figure_references(self, text: str) -> List[str]:
        """Extract figure references and captions."""
        figure_refs = []
        
        # Look for figure references
        fig_patterns = [
            r'Figure\s+\d+[^.]*?[.!]',
            r'Fig\.\s+\d+[^.]*?[.!]',
            r'(?:see|shown in|as in)\s+(?:Figure|Fig\.)\s+\d+'
        ]
        
        for pattern in fig_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            figure_refs.extend([match.strip() for match in matches])
        
        return list(set(figure_refs))[:5]  # Return top 5 unique