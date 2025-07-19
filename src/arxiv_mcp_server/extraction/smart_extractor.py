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
    """Check if NOUGAT is available."""
    try:
        import subprocess
        result = subprocess.run(["nougat", "--help"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def check_grobid_available(server_url: str = "http://localhost:8070") -> bool:
    """Check if GROBID server is available."""
    try:
        import requests
        response = requests.get(f"{server_url}/api/health", timeout=5)
        return response.status_code == 200
    except Exception:
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
        """Extract sample text from first few pages for analysis."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                sample_pages = min(len(pdf.pages), max_pages)
                text_parts = []
                
                for i in range(sample_pages):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Sample text extraction failed: {e}")
            return ""
    
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
        model: str = "mistral-ocr-latest"
    ) -> Dict[str, Any]:
        """Process document with Mistral OCR using correct API format."""
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
            
            # Prepare request payload
            payload = {
                "model": model,
                "document": {
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
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
                    
                    # Extract and combine content from all pages
                    pages = result.get("pages", [])
                    combined_content = []
                    images = []
                    
                    for page in pages:
                        page_markdown = page.get("markdown", "")
                        combined_content.append(f"# Page {page.get('index', 1)}\n\n{page_markdown}")
                        
                        # Extract images if present
                        page_images = page.get("images", [])
                        images.extend(page_images)
                    
                    full_content = "\n\n".join(combined_content)
                    
                    return {
                        "content": full_content,
                        "pages": pages,
                        "images": images,
                        "metadata": {
                            "page_count": len(pages),
                            "extraction_model": model,
                            "include_images": include_images
                        },
                        "extraction_method": "mistral_ocr_v2",
                        "quality_score": 0.95  # Mistral OCR is high quality
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Mistral OCR API error {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Mistral OCR processing failed: {e}")
            raise


class NOUGATExtractor:
    """NOUGAT-based PDF extraction for academic documents."""
    
    def __init__(self):
        self.model_tag = "0.1.17-base"  # Updated to current stable NOUGAT model
        
    async def extract_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract document using NOUGAT neural OCR."""
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Run NOUGAT command
                cmd = [
                    "nougat",
                    str(pdf_path),
                    "--out", str(output_dir),
                    "--model", self.model_tag,
                    "--no-skipping"  # Process all pages
                ]
                
                # Execute NOUGAT
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    raise Exception(f"NOUGAT failed: {stderr.decode()}")
                
                # Read the generated .mmd file
                mmd_file = output_dir / f"{pdf_path.stem}.mmd"
                if mmd_file.exists():
                    content = mmd_file.read_text(encoding='utf-8')
                    
                    return {
                        "content": content,
                        "sections": self._parse_sections(content),
                        "extraction_method": "nougat_neural_ocr",
                        "processing_time": "~10s",
                        "quality_estimate": 0.90,
                        "format": "mathpix_markdown"
                    }
                else:
                    raise Exception("NOUGAT output file not found")
                    
        except Exception as e:
            logger.error(f"NOUGAT extraction failed: {e}")
            raise
    
    def _parse_sections(self, content: str) -> Dict[str, str]:
        """Parse sections from NOUGAT markdown output."""
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


class GROBIDExtractor:
    """GROBID-based PDF extraction for academic documents."""
    
    def __init__(self, grobid_server: Optional[str] = None):
        if not GROBID_AVAILABLE:
            raise ImportError("GROBID client not available. Install with: pip install grobid-client-python")
        self.grobid_server = grobid_server or os.getenv("GROBID_SERVER", "http://localhost:8070")
        self.client = GrobidClient(grobid_server=self.grobid_server)
        
    async def extract_document(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract document using GROBID."""
        try:
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Process with GROBID (blocking call)
                result = await asyncio.to_thread(
                    self.client.process_pdf,
                    "processFulltextDocument", 
                    str(pdf_path),
                    generateIDs=True,
                    consolidate_header=True,
                    consolidate_citations=True,
                    include_raw_citations=True,
                    include_raw_affiliations=True,
                    tei_coordinates=True,
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
        self.nougat_extractor = NOUGATExtractor()
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
        
        # Step 1: Analyze difficulty (unless user forces a tier)
        if user_preference and not force_analysis:
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
            f"Success: {result['success']}"
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
                return await self._extract_with_tier(pdf_path, ExtractionTier.FAST)
            else:
                return {"error": f"All extraction methods failed: {e}"}
    
    async def _extract_fast(self, pdf_path: Path) -> Dict[str, Any]:
        """Fast extraction using current pdfplumber + PyPDF2 method."""
        content = await self.paper_reader.download_and_read_paper(
            str(pdf_path.stem), 
            format_type="pdf"
        )
        
        return {
            "content": content["content"].get("clean_text", ""),
            "sections": content["content"].get("sections", {}),
            "metadata": content["content"].get("metadata", {}),
            "extraction_method": "fast_pdfplumber",
            "processing_time": "~1s",
            "quality_estimate": 0.7
        }
    
    async def _extract_smart(self, pdf_path: Path) -> Dict[str, Any]:
        """Smart extraction using NOUGAT/GROBID with fallback."""
        # Try NOUGAT first (best for academic papers with math)
        try:
            logger.info("Attempting NOUGAT extraction...")
            return await self.nougat_extractor.extract_document(pdf_path)
        except Exception as nougat_error:
            logger.warning(f"NOUGAT failed: {nougat_error}")
            
            # Fallback to GROBID
            try:
                logger.info("Falling back to GROBID extraction...")
                if not self.grobid_extractor:
                    if GROBID_AVAILABLE:
                        self.grobid_extractor = GROBIDExtractor()
                    else:
                        raise ImportError("GROBID not available")
                return await self.grobid_extractor.extract_document(pdf_path)
            except Exception as grobid_error:
                logger.warning(f"GROBID failed: {grobid_error}")
                
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
                    "extraction_method": "smart_fallback_enhanced",
                    "processing_time": "~5s",
                    "quality_estimate": 0.75,
                    "fallback_reason": f"NOUGAT: {nougat_error}, GROBID: {grobid_error}"
                }
    
    async def _extract_premium(self, pdf_path: Path) -> Dict[str, Any]:
        """Premium extraction using Mistral OCR."""
        if not self.mistral_client:
            self.mistral_client = MistralOCRClient()
        
        async with self.mistral_client as client:
            result = await client.process_document(
                pdf_path,
                include_images=True,
                model="mistral-ocr-latest"
            )
            
            return {
                "content": result["content"],
                "pages": result["pages"],
                "images": result["images"],
                "metadata": result["metadata"],
                "extraction_method": result["extraction_method"],
                "processing_time": "~10s",
                "quality_estimate": result["quality_score"]
            }