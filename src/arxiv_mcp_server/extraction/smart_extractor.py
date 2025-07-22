"""Smart PDF extraction with three-fold adaptive mechanism."""

import os
import re
import logging
import asyncio
import subprocess
import tempfile
import json
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path

import aiohttp
import PyPDF2
import pdfplumber

try:
    from grobid_client.grobid_client import GrobidClient
    GROBID_AVAILABLE = True
except ImportError:
    GrobidClient = None
    GROBID_AVAILABLE = False

from .paper_reader import PaperReader, PaperAnalyzer

logger = logging.getLogger(__name__)


def check_nougat_available() -> bool:
    """Check if NOUGAT is available (CLI or Docker)."""
    # First try CLI
    try:
        import subprocess
        result = subprocess.run(["nougat", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.debug("Nougat CLI available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Nougat CLI not available: {e}")
    
    # Then try Docker
    try:
        result = subprocess.run(["docker", "ps", "--filter", "name=nougat-server", "--format", "{{.Names}}"],
                              capture_output=True, text=True, timeout=5)
        if "nougat-server" in result.stdout:
            logger.debug("Nougat Docker container available")
            return True
    except Exception as e:
        logger.debug(f"Nougat Docker not available: {e}")
    
    # Finally check if Docker itself is available for potential setup
    try:
        result = subprocess.run(["docker", "--version"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.debug("Docker available for Nougat setup")
            return True
    except Exception:
        pass
    
    return False


def check_grobid_available(server_url: str = "http://localhost:8070") -> bool:
    """Check if GROBID server is available."""
    try:
        import requests
        # Try multiple health endpoints
        endpoints = ["/api/health", "/api/isalive", "/"]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{server_url}{endpoint}", timeout=2)
                if response.status_code in [200, 302]:  # 302 is redirect, still healthy
                    logger.debug(f"GROBID health check successful at {server_url}{endpoint}")
                    return True
            except Exception as e:
                logger.debug(f"GROBID health check failed at {server_url}{endpoint}: {e}")
                continue
        
        logger.warning(f"GROBID server not available at {server_url}")
        return False
    except Exception as e:
        logger.error(f"GROBID availability check error: {e}")
        return False


class ExtractionTier(Enum):
    """PDF extraction complexity tiers."""
    FAST = "fast"           # pdfplumber + PyPDF2 (simple papers)
    SMART = "smart"         # NOUGAT/GROBID (moderate complexity)  
    PREMIUM = "premium"     # Mistral OCR (math/complex layouts)


class DifficultyClassifier:
    """Analyzes PDF complexity to determine optimal extraction method."""
    
    def __init__(self):
        self.math_patterns = [
            r'\$.*?\$',           # Inline math
            r'\$\$.*?\$\$',       # Display math
            r'\\begin\{equation\}', r'\\begin\{align\}',
            r'∫', r'∑', r'∏', r'√', r'∂', r'∇',  # Math symbols
            r'α', r'β', r'γ', r'δ', r'θ', r'λ', r'μ', r'π', r'σ', r'φ'  # Greek letters
        ]
        
        self.complex_layout_patterns = [
            r'Figure\s+\d+', r'Table\s+\d+',  # Figures/tables
            r'\btwo.?column\b', r'\bmulti.?column\b',  # Multi-column
            r'\\includegraphics', r'\\begin\{figure\}', r'\\begin\{table\}'  # LaTeX elements
        ]
    
    async def analyze_difficulty(self, pdf_path: Path) -> Dict[str, Any]:
        """Analyze PDF to determine extraction difficulty."""
        analysis = {
            "tier_recommendation": ExtractionTier.FAST,
            "confidence": 0.0,
            "factors": {
                "page_count": 0,
                "math_density": 0.0,
                "layout_complexity": 0.0,
                "text_extractability": 0.0,
                "file_size_mb": 0.0
            },
            "reasoning": []
        }
        
        try:
            # Basic file metrics
            file_size = pdf_path.stat().st_size / (1024 * 1024)  # MB
            analysis["factors"]["file_size_mb"] = file_size
            
            # Quick text extraction for analysis
            sample_text = await self._extract_sample_text(pdf_path)
            
            if not sample_text:
                analysis["tier_recommendation"] = ExtractionTier.PREMIUM
                analysis["reasoning"].append("No extractable text - likely scanned/image PDF")
                analysis["confidence"] = 0.9
                return analysis
            
            # Analyze content
            page_count = await self._get_page_count(pdf_path)
            math_density = self._calculate_math_density(sample_text)
            layout_complexity = self._calculate_layout_complexity(sample_text)
            text_extractability = self._calculate_text_extractability(sample_text)
            
            analysis["factors"].update({
                "page_count": page_count,
                "math_density": math_density,
                "layout_complexity": layout_complexity,
                "text_extractability": text_extractability
            })
            
            # Decision logic
            tier, confidence, reasoning = self._determine_tier(analysis["factors"])
            analysis["tier_recommendation"] = tier
            analysis["confidence"] = confidence
            analysis["reasoning"] = reasoning
            
        except Exception as e:
            logger.error(f"Error analyzing PDF difficulty: {e}")
            analysis["tier_recommendation"] = ExtractionTier.FAST
            analysis["reasoning"].append(f"Analysis failed, using fallback: {e}")
            analysis["confidence"] = 0.1
        
        return analysis
    
    async def _extract_sample_text(self, pdf_path: Path, max_pages: int = 3) -> str:
        """Extract sample text from strategically selected pages for analysis."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    return ""
                
                # Strategy: Sample from beginning, middle, and end for better representation
                sample_indices = self._get_representative_page_indices(total_pages, max_pages)
                
                text_parts = []
                for i in sample_indices:
                    try:
                        page_text = pdf.pages[i].extract_text()
                        if page_text:
                            text_parts.append(f"[Page {i+1}]\n{page_text}")
                    except Exception as e:
                        logger.debug(f"Failed to extract text from page {i+1}: {e}")
                        continue
                
                return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Sample text extraction failed: {e}")
            return ""
    
    def _get_representative_page_indices(self, total_pages: int, max_samples: int = 3) -> list[int]:
        """Get representative page indices for better complexity assessment."""
        import random
        
        if total_pages <= max_samples:
            return list(range(total_pages))
        
        # Strategy depends on document length
        if total_pages <= 5:
            # Short docs: sample all or most pages
            return list(range(min(total_pages, max_samples)))
        
        elif total_pages <= 15:
            # Medium docs: beginning, middle, end
            indices = [
                0,  # First page (abstract/intro)
                total_pages // 2,  # Middle (methods/results)
                total_pages - 1   # End (conclusions)
            ]
            return indices[:max_samples]
        
        else:
            # Long docs: strategic sampling with randomization
            # Always include: early (but not first), middle, and late sections
            sections = [
                range(1, min(4, total_pages)),           # Early (skip title page)
                range(total_pages // 3, 2 * total_pages // 3),  # Middle (methods/results)
                range(2 * total_pages // 3, total_pages - 1)    # Late (conclusions)
            ]
            
            indices = []
            samples_per_section = max(1, max_samples // len(sections))
            
            for section in sections:
                if section and len(indices) < max_samples:
                    # Randomly sample from each section
                    section_pages = list(section)
                    sample_count = min(samples_per_section, len(section_pages), max_samples - len(indices))
                    if sample_count > 0:
                        indices.extend(random.sample(section_pages, sample_count))
            
            # Fill remaining slots with random pages if needed
            while len(indices) < max_samples and len(indices) < total_pages:
                remaining_pages = [i for i in range(total_pages) if i not in indices]
                if remaining_pages:
                    indices.append(random.choice(remaining_pages))
                else:
                    break
            
            return sorted(indices[:max_samples])
    
    async def _get_page_count(self, pdf_path: Path) -> int:
        """Get total page count."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception:
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    return len(reader.pages)
            except Exception:
                return 0
    
    def _calculate_math_density(self, text: str) -> float:
        """Calculate mathematical content density (0-1)."""
        if not text:
            return 0.0
        
        math_matches = 0
        for pattern in self.math_patterns:
            math_matches += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Normalize by text length
        text_length = len(text.split())
        return min(math_matches / max(text_length, 1) * 100, 1.0)
    
    def _calculate_layout_complexity(self, text: str) -> float:
        """Calculate layout complexity score (0-1)."""
        if not text:
            return 0.0
        
        complexity_indicators = 0
        
        # Check for complex layout patterns
        for pattern in self.complex_layout_patterns:
            complexity_indicators += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Check for multi-column indicators
        lines = text.split('\n')
        short_lines = sum(1 for line in lines if len(line.strip()) < 50)
        if len(lines) > 0:
            short_line_ratio = short_lines / len(lines)
            if short_line_ratio > 0.3:  # Many short lines suggest multi-column
                complexity_indicators += 5
        
        # Normalize complexity score
        return min(complexity_indicators / 10.0, 1.0)
    
    def _calculate_text_extractability(self, text: str) -> float:
        """Calculate how well text extracts (0-1, higher = better extraction)."""
        if not text:
            return 0.0
        
        # Check for extraction artifacts
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Count problematic characters
        garbled_chars = text.count('�')  # Replacement character
        weird_chars = len(re.findall(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', text))
        
        # Calculate extraction quality
        problem_ratio = (garbled_chars + weird_chars) / total_chars
        return max(0.0, 1.0 - problem_ratio * 10)  # Amplify problems
    
    def _determine_tier(self, factors: Dict[str, float]) -> tuple[ExtractionTier, float, List[str]]:
        """Determine optimal extraction tier based on analysis factors."""
        reasoning = []
        score = 0.0
        
        # Page count factor
        if factors["page_count"] > 50:
            score += 0.3
            reasoning.append(f"Large document ({factors['page_count']} pages)")
        elif factors["page_count"] > 20:
            score += 0.1
            reasoning.append(f"Medium document ({factors['page_count']} pages)")
        
        # Math density factor
        if factors["math_density"] > 0.1:
            score += 0.4
            reasoning.append(f"High math density ({factors['math_density']:.1%})")
        elif factors["math_density"] > 0.05:
            score += 0.2
            reasoning.append(f"Moderate math content ({factors['math_density']:.1%})")
        
        # Layout complexity factor
        if factors["layout_complexity"] > 0.3:
            score += 0.3
            reasoning.append(f"Complex layout detected ({factors['layout_complexity']:.1%})")
        elif factors["layout_complexity"] > 0.1:
            score += 0.1
            reasoning.append(f"Some layout complexity ({factors['layout_complexity']:.1%})")
        
        # Text extractability factor
        if factors["text_extractability"] < 0.5:
            score += 0.5
            reasoning.append(f"Poor text extraction ({factors['text_extractability']:.1%})")
        elif factors["text_extractability"] < 0.8:
            score += 0.2
            reasoning.append(f"Moderate extraction issues ({factors['text_extractability']:.1%})")
        
        # File size factor
        if factors["file_size_mb"] > 10:
            score += 0.2
            reasoning.append(f"Large file size ({factors['file_size_mb']:.1f}MB)")
        
        # Determine tier based on score
        if score >= 0.6:
            return ExtractionTier.PREMIUM, min(score, 0.95), reasoning
        elif score >= 0.3:
            return ExtractionTier.SMART, score * 0.8, reasoning
        else:
            return ExtractionTier.FAST, max(0.7 - score, 0.1), reasoning


class MistralOCRClient:
    """Client for Mistral OCR API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1/ocr"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_document(
        self, 
        pdf_path: Path,
        include_images: bool = True,
        model: str = "mistral-ocr-latest",
        academic_mode: bool = True
    ) -> Dict[str, Any]:
        """Process document with Mistral OCR optimized for academic papers."""
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Convert PDF to base64 for API
            import base64
            with open(pdf_path, 'rb') as file:
                pdf_content = file.read()
                pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
            
            # Prepare request payload (corrected API structure)
            payload = {
                "document": {
                    "type": "document_url", 
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                "model": "mistral-ocr-latest",
                "include_image_base64": include_images
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with self.session.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Enhanced processing for academic papers
                    pages = result.get("pages", [])
                    combined_content = []
                    images = []
                    sections = {}
                    
                    # Process each page with academic structure awareness
                    for page in pages:
                        page_markdown = page.get("markdown", "")
                        page_num = page.get('index', 1)
                        
                        # Don't add page headers for academic papers - preserve natural flow
                        if academic_mode and page_num > 1:
                            combined_content.append(page_markdown)
                        else:
                            combined_content.append(page_markdown)
                        
                        # Extract images if present
                        page_images = page.get("images", [])
                        images.extend(page_images)
                    
                    full_content = "\n\n".join(combined_content)
                    
                    # Parse academic sections if in academic mode
                    if academic_mode:
                        sections = self._parse_academic_sections(full_content)
                    
                    # Calculate quality score based on content analysis
                    quality_score = self._calculate_mistral_quality(full_content, result, academic_mode)
                    
                    return {
                        "content": full_content,
                        "sections": sections,
                        "pages": pages,
                        "images": images,
                        "metadata": {
                            "page_count": len(pages),
                            "extraction_model": model,
                            "include_images": include_images,
                            "academic_mode": academic_mode,
                            "math_equations_found": self._count_math_content(full_content),
                            "table_count": full_content.count("|")//4 if full_content else 0
                        },
                        "extraction_method": f"mistral_ocr_{'academic' if academic_mode else 'standard'}",
                        "quality_score": quality_score
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Mistral OCR API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Mistral OCR processing failed: {e}")
            raise
    
    def _parse_academic_sections(self, content: str) -> Dict[str, str]:
        """Parse academic paper sections from Mistral OCR output."""
        sections = {}
        if not content:
            return sections
            
        current_section = "abstract"
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Detect academic section headers
            if line.lower().startswith('#'):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Extract new section name
                section_title = line.lstrip('#').strip().lower()
                
                # Map to common academic sections
                if 'abstract' in section_title:
                    current_section = 'abstract'
                elif any(word in section_title for word in ['introduction', 'intro']):
                    current_section = 'introduction'
                elif any(word in section_title for word in ['method', 'approach', 'technique']):
                    current_section = 'methodology'
                elif any(word in section_title for word in ['result', 'finding', 'experiment']):
                    current_section = 'results'
                elif any(word in section_title for word in ['discussion', 'analysis']):
                    current_section = 'discussion'
                elif any(word in section_title for word in ['conclusion', 'summary']):
                    current_section = 'conclusion'
                elif any(word in section_title for word in ['reference', 'bibliography']):
                    current_section = 'references'
                else:
                    current_section = section_title.replace(' ', '_')
                
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _calculate_mistral_quality(self, content: str, api_result: dict, academic_mode: bool) -> float:
        """Calculate quality score based on content analysis."""
        if not content:
            return 0.3
        
        base_score = 0.95  # Mistral OCR baseline
        
        # Adjust based on content characteristics
        math_content = self._count_math_content(content)
        table_indicators = content.count("|") // 4
        
        if academic_mode:
            # Bonus for academic content preservation
            if math_content > 0:
                base_score = min(0.98, base_score + 0.02)
            if table_indicators > 0:
                base_score = min(0.97, base_score + 0.01)
            if any(section in content.lower() for section in ['abstract', 'introduction', 'methodology']):
                base_score = min(0.96, base_score + 0.01)
        
        # Check for potential quality issues
        if len(content) < 500:
            base_score *= 0.8  # Short content might be incomplete
        
        return round(base_score, 2)
    
    def _count_math_content(self, content: str) -> int:
        """Count mathematical content indicators."""
        if not content:
            return 0
            
        math_indicators = [
            '$$', '$', '\\frac', '\\sum', '\\int', '\\alpha', '\\beta', '\\gamma',
            '\\theta', '\\sigma', '\\pi', '\\mu', '\\lambda', '\\delta', '\\epsilon',
            '\\begin{equation}', '\\end{equation}', '\\begin{align}', '\\end{align}',
            '\\mathbf', '\\mathit', '\\mathrm', '\\partial', '\\nabla'
        ]
        
        count = 0
        for indicator in math_indicators:
            count += content.count(indicator)
        
        return count



class GROBIDExtractor:
    """GROBID-based PDF extraction for academic documents."""
    
    def __init__(self, grobid_server: Optional[str] = None):
        if not GROBID_AVAILABLE:
            raise ImportError("GROBID client not available. Install with: pip install grobid-client-python")
        self.grobid_server = grobid_server or os.getenv("GROBID_SERVER", "http://localhost:8070")
        
        # Configure GROBID client with longer timeout for academic papers
        self.client = GrobidClient(grobid_server=self.grobid_server)
        # Override timeout settings for academic paper processing
        self.client.config = {
            "grobid_server": self.grobid_server,
            "batch_size": 1000,
            "sleep_time": 10,  # Wait longer between retries (was 5)
            "timeout": 180,    # Increase timeout to 3 minutes (was 60)
            "coordinates": ["persName", "figure", "ref", "biblStruct", "formula", "s"]
        }
        
    async def extract_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract document using GROBID."""
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Process with GROBID (blocking call with simplified parameters)
                result = await asyncio.to_thread(
                    self.client.process_pdf,
                    "processFulltextDocument", 
                    str(pdf_path),
                    generateIDs=False,  # Simplified to avoid timeout
                    consolidate_header=False,  # Simplified to avoid timeout
                    consolidate_citations=False,  # Simplified to avoid timeout
                    include_raw_citations=False,
                    include_raw_affiliations=False,
                    tei_coordinates=False,
                    segment_sentences=False
                )
                
                # GROBID client returns (status, status_code, xml_content)
                if isinstance(result, tuple) and len(result) >= 3:
                    status, status_code, xml_content = result
                    if xml_content and status_code == 200:
                        # Parse TEI XML into structured content
                        parsed_content = self._parse_tei_xml(xml_content)
                        
                        return {
                            "content": parsed_content["text"],
                            "sections": parsed_content["sections"],
                            "metadata": parsed_content["metadata"],
                            "references": parsed_content["references"],
                            "extraction_method": "grobid_tei",
                            "processing_time": "~5s",
                            "quality_estimate": 0.85,
                            "format": "structured_tei"
                        }
                    else:
                        raise Exception(f"GROBID processing failed: status={status}, code={status_code}")
                else:
                    raise Exception("GROBID returned invalid response format")
                    
        except Exception as e:
            logger.error(f"GROBID extraction failed: {e}")
            raise
    
    def _parse_tei_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse GROBID TEI XML output into structured content."""
        try:
            from xml.etree import ElementTree as ET
            
            root = ET.fromstring(xml_content)
            
            # Define TEI namespace
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            
            # Extract metadata
            metadata = {}
            title_elem = root.find('.//tei:title[@type="main"]', ns)
            if title_elem is not None:
                metadata['title'] = title_elem.text or ""
            
            # Extract authors
            authors = []
            for author in root.findall('.//tei:author', ns):
                name_parts = []
                for name_part in author.findall('.//tei:forename', ns):
                    if name_part.text:
                        name_parts.append(name_part.text)
                surname = author.find('.//tei:surname', ns)
                if surname is not None and surname.text:
                    name_parts.append(surname.text)
                if name_parts:
                    authors.append(' '.join(name_parts))
            metadata['authors'] = authors
            
            # Extract abstract
            abstract_elem = root.find('.//tei:abstract', ns)
            abstract = ""
            if abstract_elem is not None:
                abstract = ET.tostring(abstract_elem, encoding='unicode', method='text')
            
            # Extract sections
            sections = {"abstract": abstract}
            body = root.find('.//tei:body', ns)
            if body is not None:
                for div in body.findall('.//tei:div', ns):
                    head = div.find('.//tei:head', ns)
                    if head is not None and head.text:
                        section_name = head.text.lower().replace(' ', '_')
                        section_text = ET.tostring(div, encoding='unicode', method='text')
                        sections[section_name] = section_text
            
            # Extract references
            references = []
            for ref in root.findall('.//tei:biblStruct', ns):
                ref_text = ET.tostring(ref, encoding='unicode', method='text')
                if ref_text.strip():
                    references.append(ref_text.strip())
            
            # Combine all text
            full_text = ET.tostring(root, encoding='unicode', method='text')
            
            return {
                "text": full_text,
                "sections": sections,
                "metadata": metadata,
                "references": references
            }
            
        except Exception as e:
            logger.warning(f"TEI XML parsing failed: {e}")
            return {
                "text": xml_content,
                "sections": {},
                "metadata": {},
                "references": []
            }


class SmartPDFExtractor:
    """Intelligent PDF extractor with three-tier adaptive mechanism."""
    
    def __init__(self):
        self.classifier = DifficultyClassifier()
        self.paper_reader = PaperReader()
        self.paper_analyzer = PaperAnalyzer()
        self.mistral_client = None
        self.grobid_extractor = None  # Will be initialized when needed
    
    async def extract_paper(
        self,
        pdf_path: Path,
        user_preference: Optional[ExtractionTier] = None,
        budget_mode: bool = False,
        force_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Extract paper content using adaptive three-tier mechanism.
        
        Args:
            pdf_path: Path to PDF file
            user_preference: Force specific extraction tier
            budget_mode: If True, avoid paid services
            force_analysis: If True, always run difficulty analysis
        """
        
        # Step 1: Check for FORCE_SMART environment variable for academic papers
        force_smart = os.getenv("FORCE_SMART", "false").lower() == "true"
        if force_smart and not user_preference:
            analysis = {
                "tier_recommendation": ExtractionTier.SMART,
                "confidence": 1.0,
                "reasoning": ["Forced SMART tier for academic papers (FORCE_SMART=true)"]
            }
            logger.info("FORCE_SMART enabled - using SMART tier for academic paper")
        # Step 2: Analyze difficulty (unless user forces a tier)
        elif user_preference and not force_analysis:
            analysis = {
                "tier_recommendation": user_preference,
                "confidence": 1.0,
                "reasoning": ["User-specified tier"]
            }
        else:
            analysis = await self.classifier.analyze_difficulty(pdf_path)
            
            # Override with user preference if provided
            if user_preference:
                analysis["tier_recommendation"] = user_preference
                analysis["reasoning"].append(f"Overridden by user preference: {user_preference.value}")
        
        # Step 2: Apply budget constraints
        if budget_mode and analysis["tier_recommendation"] == ExtractionTier.PREMIUM:
            analysis["tier_recommendation"] = ExtractionTier.SMART
            analysis["reasoning"].append("Downgraded from PREMIUM to SMART due to budget mode")
        
        # Step 3: Extract using appropriate method
        extraction_result = await self._extract_with_tier(
            pdf_path, 
            analysis["tier_recommendation"]
        )
        
        # Step 4: Combine results
        result = {
            "pdf_path": str(pdf_path),
            "analysis": analysis,
            "extraction": extraction_result,
            "tier_used": analysis["tier_recommendation"].value,
            "success": "error" not in extraction_result
        }
        
        logger.info(
            f"PDF extraction complete - Tier: {analysis['tier_recommendation'].value}, "
            f"Confidence: {analysis['confidence']:.2f}, "
            f"Success: {result['success']}, "
            f"Method: {extraction_result.get('extraction_method', 'unknown')}, "
            f"Processing: {extraction_result.get('processing_time', 'unknown')}"
        )
        
        return result
    
    async def _extract_with_tier(self, pdf_path: Path, tier: ExtractionTier) -> Dict[str, Any]:
        """Extract content using the specified tier method."""
        
        try:
            if tier == ExtractionTier.FAST:
                return await self._extract_fast(pdf_path)
            elif tier == ExtractionTier.SMART:
                return await self._extract_smart(pdf_path)
            elif tier == ExtractionTier.PREMIUM:
                return await self._extract_premium(pdf_path)
            else:
                raise ValueError(f"Unknown extraction tier: {tier}")
                
        except Exception as e:
            logger.error(f"Extraction failed for tier {tier.value}: {e}")
            
            # Fallback to simpler methods
            if tier == ExtractionTier.PREMIUM:
                logger.info("Falling back from PREMIUM to SMART")
                return await self._extract_with_tier(pdf_path, ExtractionTier.SMART)
            elif tier == ExtractionTier.SMART:
                logger.info("Falling back from SMART to FAST")
                fallback_result = await self._extract_with_tier(pdf_path, ExtractionTier.FAST)
                
                # Add fallback information to the result
                if isinstance(fallback_result, dict) and not fallback_result.get("error"):
                    user_message = f"⚠️ SMART extraction failed ({str(e)[:100]}{'...' if len(str(e)) > 100 else ''}). Falling back to fast extraction for reliable text processing."
                    
                    fallback_result["fallback_info"] = {
                        "attempted_method": "smart_extraction",
                        "attempted_tier": tier.value,
                        "fallback_reason": str(e),
                        "fallback_to": "fast_pdfplumber_direct",
                        "user_message": user_message
                    }
                    fallback_result["extraction_method"] = f"fast_fallback_from_{tier.value}"
                    fallback_result["processing_note"] = "Fallback used - see fallback_info for details"
                
                return fallback_result
            else:
                return {"error": f"All extraction methods failed: {e}"}
    
    async def _extract_fast(self, pdf_path: Path) -> Dict[str, Any]:
        """Fast extraction using direct pdfplumber + PyPDF2 method."""
        try:
            # Direct file extraction without using paper_reader
            content = await self._extract_with_pdfplumber(pdf_path)
            
            return {
                "content": content.get("text", ""),
                "sections": content.get("sections", {}),
                "metadata": content.get("metadata", {}),
                "extraction_method": "fast_pdfplumber_direct",
                "processing_time": "~1s",
                "quality_estimate": 0.7
            }
        except Exception as e:
            logger.error(f"Fast extraction failed: {e}")
            raise
    
    async def _extract_with_pdfplumber(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract content directly using pdfplumber."""
        text_parts = []
        sections = {}
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata = {
                    "page_count": len(pdf.pages),
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "creator": pdf.metadata.get("Creator", "")
                }
                
                current_section = "content"
                section_content = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Simple section detection
                            lines = page_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and len(line) < 100 and any(keyword in line.lower() for keyword in 
                                    ['abstract', 'introduction', 'method', 'result', 'conclusion', 'reference']):
                                    # Likely a section header
                                    if section_content:
                                        sections[current_section] = '\n'.join(section_content)
                                    current_section = line.lower().replace(' ', '_')
                                    section_content = []
                                else:
                                    section_content.append(line)
                            
                            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                # Save last section
                if section_content:
                    sections[current_section] = '\n'.join(section_content)
                
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        except Exception as e:
                            logger.warning(f"PyPDF2 failed on page {page_num + 1}: {e}")
                            continue
            except Exception as fallback_error:
                logger.error(f"PyPDF2 fallback also failed: {fallback_error}")
                raise Exception(f"All fast extraction methods failed: {e}, {fallback_error}")
        
        full_text = '\n\n'.join(text_parts)
        
        return {
            "text": full_text,
            "sections": sections,
            "metadata": metadata
        }
    
    async def _extract_smart(self, pdf_path: Path) -> Dict[str, Any]:
        """Smart extraction with hybrid GROBID + Mistral approach for academic papers."""
        force_smart = os.getenv("FORCE_SMART", "false").lower() == "true"
        enable_mistral_enhancement = os.getenv("MISTRAL_ENHANCEMENT", "true").lower() == "true"
        
        # Try GROBID first (reliable structure extraction)
        grobid_server_url = os.getenv("GROBID_SERVER", "http://localhost:8070")
        grobid_result = None
        
        if force_smart or check_grobid_available(grobid_server_url):
            try:
                logger.info(f"Attempting GROBID extraction at {grobid_server_url}...")
                if not self.grobid_extractor:
                    if GROBID_AVAILABLE:
                        self.grobid_extractor = GROBIDExtractor(grobid_server_url)
                    else:
                        raise ImportError("GROBID client not available")
                
                grobid_result = await self.grobid_extractor.extract_document(pdf_path)
                
                # Enhance with Mistral for mathematical content if available and enabled
                if enable_mistral_enhancement and os.getenv("MISTRAL_API_KEY"):
                    try:
                        logger.info("Enhancing GROBID result with Mistral for mathematical content...")
                        enhanced_result = await self._enhance_with_mistral(pdf_path, grobid_result)
                        return enhanced_result
                    except Exception as mistral_error:
                        logger.warning(f"Mistral enhancement failed: {mistral_error}, using GROBID result")
                
                return grobid_result
                
            except Exception as grobid_error:
                logger.warning(f"GROBID extraction failed: {grobid_error}")
                if force_smart:
                    # Don't fallback if explicitly forced
                    raise grobid_error
        else:
            logger.warning(f"GROBID server not available at {grobid_server_url}")
        
        # Try pure Mistral as fallback if GROBID failed
        if enable_mistral_enhancement and os.getenv("MISTRAL_API_KEY"):
            try:
                logger.info("Using pure Mistral extraction as SMART tier fallback...")
                if not self.mistral_client:
                    self.mistral_client = MistralOCRClient()
                
                async with self.mistral_client as client:
                    result = await client.process_document(pdf_path, academic_mode=True)
                    result["extraction_method"] = "mistral_smart_fallback"
                    result["processing_time"] = "~8s"
                    return result
                    
            except Exception as mistral_error:
                logger.warning(f"Mistral fallback failed: {mistral_error}")
        
        # Final fallback to enhanced basic extraction
        logger.info("Using enhanced basic extraction as final fallback...")
        content = await self.paper_reader.download_and_read_paper(
            str(pdf_path.stem),
            format_type="pdf"
        )
        
        # Enhanced processing for SMART tier
        summary = await self.paper_analyzer.summarize_paper(str(pdf_path.stem))
        findings = await self.paper_analyzer.extract_key_findings(str(pdf_path.stem))
        
        return {
            "content": content["content"].get("clean_text", ""),
            "sections": content["content"].get("sections", {}),
            "summary": summary,
            "key_findings": findings,
            "metadata": content["content"].get("metadata", {}),
            "extraction_method": "smart_enhanced_fallback",
            "processing_time": "~5s",
            "quality_estimate": 0.75,
            "fallback_info": {
                "fallback_reason": "GROBID and Mistral unavailable or failed",
                "user_message": "⚠️ Advanced extraction methods failed. Using enhanced basic extraction for reliable text processing.",
                "final_method": "enhanced_basic_extraction"
            }
        }
    
    async def _enhance_with_mistral(self, pdf_path: Path, grobid_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance GROBID extraction with Mistral for mathematical content."""
        if not self.mistral_client:
            self.mistral_client = MistralOCRClient()
        
        try:
            # Get Mistral extraction focused on mathematical content
            async with self.mistral_client as client:
                mistral_result = await client.process_document(
                    pdf_path, 
                    academic_mode=True,
                    include_images=False  # Focus on text/math for enhancement
                )
            
            # Combine GROBID structure with Mistral mathematical content
            enhanced_result = grobid_result.copy()
            
            # Analyze mathematical content density
            grobid_math_score = self.mistral_client._count_math_content(grobid_result.get("content", ""))
            mistral_math_score = self.mistral_client._count_math_content(mistral_result.get("content", ""))
            
            # If Mistral found significantly more math content, prefer its content
            if mistral_math_score > grobid_math_score * 1.5:
                logger.info(f"Using Mistral content (math score: {mistral_math_score} vs GROBID: {grobid_math_score})")
                enhanced_result["content"] = mistral_result["content"]
                enhanced_result["sections"] = mistral_result.get("sections", enhanced_result.get("sections", {}))
                enhanced_result["quality_estimate"] = 0.92  # High quality hybrid
            else:
                # Use GROBID structure but enhance mathematical sections
                enhanced_result["quality_estimate"] = 0.88  # Good quality enhanced
                logger.info("Using GROBID structure with mathematical content validation")
            
            # Combine metadata
            enhanced_result["metadata"] = {
                **enhanced_result.get("metadata", {}),
                "enhancement_method": "grobid_mistral_hybrid",
                "mistral_math_score": mistral_math_score,
                "grobid_math_score": grobid_math_score,
                "math_enhancement_used": mistral_math_score > grobid_math_score * 1.5
            }
            
            # Update method and timing
            enhanced_result["extraction_method"] = "grobid_mistral_enhanced"
            enhanced_result["processing_time"] = "~8s"
            
            logger.info(f"Hybrid extraction complete - GROBID structure + Mistral math enhancement")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Mistral enhancement failed: {e}")
            # Return original GROBID result if enhancement fails
            return grobid_result
    
    async def _extract_premium(self, pdf_path: Path) -> Dict[str, Any]:
        """Premium extraction using pure Mistral OCR with maximum quality."""
        if not self.mistral_client:
            self.mistral_client = MistralOCRClient()
        
        async with self.mistral_client as client:
            result = await client.process_document(
                pdf_path,
                include_images=True,
                model="mistral-ocr-latest",
                academic_mode=True
            )
            
            # Premium tier gets the full Mistral result with images
            return {
                "content": result["content"],
                "sections": result.get("sections", {}),
                "pages": result["pages"],
                "images": result["images"],
                "metadata": result["metadata"],
                "extraction_method": result["extraction_method"],
                "processing_time": "~12s",
                "quality_estimate": result["quality_score"]
            }