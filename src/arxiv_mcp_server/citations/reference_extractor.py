"""Extract and parse references from GROBID and Mistral extraction results."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """Structured reference information."""
    raw_text: str
    title: Optional[str] = None
    authors: List[str] = None
    year: Optional[str] = None
    venue: Optional[str] = None  # Journal/conference
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    confidence: float = 0.0  # 0-1 confidence in extraction
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []


class ReferenceExtractor:
    """Extract and structure references from paper extraction results."""
    
    def __init__(self):
        # arXiv ID patterns
        self.arxiv_patterns = [
            r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)',  # arXiv:2301.07041
            r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv.org/abs/2301.07041
            r'(\d{4}\.\d{4,5})(?:\s+\[cs\.\w+\])?',  # 2301.07041 [cs.AI]
        ]
        
        # DOI patterns
        self.doi_pattern = r'10\.\d+/[^\s]+'
        
        # Year pattern
        self.year_pattern = r'\b(19\d{2}|20\d{2})\b'
        
        # Common reference separators
        self.ref_separators = [
            r'\[\d+\]',  # [1], [23]
            r'\(\d+\)',  # (1), (23)
            r'^\d+\.',   # 1., 23.
        ]
    
    def extract_from_grobid(self, grobid_result: Dict[str, Any]) -> List[Reference]:
        """Extract references from GROBID TEI XML result."""
        references = []
        
        try:
            # Check if we have XML content to parse
            xml_content = grobid_result.get('xml_content')
            if not xml_content:
                # Fallback to structured references if available
                if 'references' in grobid_result:
                    return self._parse_structured_references(grobid_result['references'])
                return references
            
            # Parse TEI XML
            root = ET.fromstring(xml_content)
            
            # Find bibliography section
            ns = {'tei': 'http://www.tei-c.org/ns/1.0'}
            biblio_refs = root.findall('.//tei:listBibl/tei:biblStruct', ns)
            
            for bibl in biblio_refs:
                ref = self._parse_grobid_biblstruct(bibl, ns)
                if ref:
                    references.append(ref)
            
            logger.info(f"Extracted {len(references)} references from GROBID TEI XML")
            
        except Exception as e:
            logger.error(f"Error extracting GROBID references: {e}")
            # Fallback to text-based extraction
            if 'content' in grobid_result:
                references = self._extract_from_text(grobid_result['content'])
        
        return references
    
    def _parse_grobid_biblstruct(self, bibl_elem: ET.Element, ns: Dict[str, str]) -> Optional[Reference]:
        """Parse a GROBID biblStruct element into a Reference."""
        try:
            ref = Reference(raw_text="")
            
            # Extract title
            title_elem = bibl_elem.find('.//tei:title[@level="a"]', ns)
            if title_elem is not None and title_elem.text:
                ref.title = title_elem.text.strip()
            
            # Extract authors
            authors = []
            for author in bibl_elem.findall('.//tei:author/tei:persName', ns):
                forename = author.find('tei:forename', ns)
                surname = author.find('tei:surname', ns)
                
                name_parts = []
                if forename is not None and forename.text:
                    name_parts.append(forename.text.strip())
                if surname is not None and surname.text:
                    name_parts.append(surname.text.strip())
                
                if name_parts:
                    authors.append(' '.join(name_parts))
            
            ref.authors = authors
            
            # Extract year
            date_elem = bibl_elem.find('.//tei:date', ns)
            if date_elem is not None and date_elem.get('when'):
                ref.year = date_elem.get('when')[:4]  # Extract year part
            
            # Extract venue (journal/conference)
            journal_elem = bibl_elem.find('.//tei:title[@level="j"]', ns)
            if journal_elem is not None and journal_elem.text:
                ref.venue = journal_elem.text.strip()
            
            # Extract DOI
            doi_elem = bibl_elem.find('.//tei:idno[@type="DOI"]', ns)
            if doi_elem is not None and doi_elem.text:
                ref.doi = doi_elem.text.strip()
            
            # Extract arXiv ID
            arxiv_elem = bibl_elem.find('.//tei:idno[@type="arXiv"]', ns)
            if arxiv_elem is not None and arxiv_elem.text:
                ref.arxiv_id = self._clean_arxiv_id(arxiv_elem.text.strip())
            
            # Build raw text representation
            raw_parts = []
            if ref.authors:
                raw_parts.append(', '.join(ref.authors))
            if ref.title:
                raw_parts.append(f'"{ref.title}"')
            if ref.venue:
                raw_parts.append(ref.venue)
            if ref.year:
                raw_parts.append(f'({ref.year})')
            
            ref.raw_text = '. '.join(raw_parts)
            ref.confidence = 0.9  # High confidence for GROBID structured data
            
            return ref if ref.title or ref.authors else None
            
        except Exception as e:
            logger.warning(f"Error parsing GROBID biblStruct: {e}")
            return None
    
    def _parse_structured_references(self, references: List[str]) -> List[Reference]:
        """Parse structured reference list from GROBID."""
        ref_objects = []
        
        for ref_text in references:
            ref = self._parse_reference_text(ref_text)
            if ref:
                ref_objects.append(ref)
        
        return ref_objects
    
    def extract_from_mistral(self, mistral_result: Dict[str, Any]) -> List[Reference]:
        """Extract references from Mistral extraction result."""
        references = []
        
        try:
            content = mistral_result.get('content', '')
            if not content:
                return references
            
            # Look for references section
            references = self._extract_from_text(content)
            logger.info(f"Extracted {len(references)} references from Mistral content")
            
        except Exception as e:
            logger.error(f"Error extracting Mistral references: {e}")
        
        return references
    
    def _extract_from_text(self, content: str) -> List[Reference]:
        """Extract references from plain text content."""
        references = []
        
        # Find references section
        ref_section = self._find_references_section(content)
        if not ref_section:
            return references
        
        # Split into individual references
        ref_lines = self._split_references(ref_section)
        
        for ref_line in ref_lines:
            ref = self._parse_reference_text(ref_line)
            if ref:
                references.append(ref)
        
        return references
    
    def _find_references_section(self, content: str) -> Optional[str]:
        """Find and extract the references section from text."""
        # Common reference section headers
        ref_headers = [
            r'\n\s*References\s*\n',
            r'\n\s*REFERENCES\s*\n',
            r'\n\s*Bibliography\s*\n',
            r'\n\s*BIBLIOGRAPHY\s*\n'
        ]
        
        for pattern in ref_headers:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Extract everything after the references header
                start = match.end()
                
                # Look for next major section (Appendix, etc.)
                end_patterns = [
                    r'\n\s*Appendix\s*\n',
                    r'\n\s*APPENDIX\s*\n',
                    r'\n\s*Index\s*\n'
                ]
                
                end = len(content)
                for end_pattern in end_patterns:
                    end_match = re.search(end_pattern, content[start:], re.IGNORECASE)
                    if end_match:
                        end = start + end_match.start()
                        break
                
                return content[start:end].strip()
        
        return None
    
    def _split_references(self, ref_section: str) -> List[str]:
        """Split references section into individual references."""
        references = []
        
        # Try different splitting strategies
        lines = ref_section.split('\n')
        current_ref = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_ref:
                    references.append('\n'.join(current_ref))
                    current_ref = []
                continue
            
            # Check if this starts a new reference
            is_new_ref = False
            for sep_pattern in self.ref_separators:
                if re.match(sep_pattern, line):
                    is_new_ref = True
                    break
            
            if is_new_ref and current_ref:
                # Save previous reference and start new one
                references.append('\n'.join(current_ref))
                current_ref = [line]
            else:
                current_ref.append(line)
        
        # Don't forget the last reference
        if current_ref:
            references.append('\n'.join(current_ref))
        
        return [ref for ref in references if len(ref.strip()) > 20]  # Filter very short refs
    
    def _parse_reference_text(self, ref_text: str) -> Optional[Reference]:
        """Parse a single reference text into structured Reference object."""
        if not ref_text or len(ref_text.strip()) < 10:
            return None
        
        ref = Reference(raw_text=ref_text.strip())
        
        # Extract arXiv ID
        ref.arxiv_id = self._extract_arxiv_id(ref_text)
        
        # Extract DOI
        doi_match = re.search(self.doi_pattern, ref_text)
        if doi_match:
            ref.doi = doi_match.group(0)
        
        # Extract year
        year_matches = re.findall(self.year_pattern, ref_text)
        if year_matches:
            ref.year = year_matches[-1]  # Usually the last year is publication year
        
        # Extract title (text in quotes or after author names)
        title_patterns = [
            r'"([^"]+)"',  # "Title in quotes"
            r"'([^']+)'",  # 'Title in single quotes'
            r'\.([^.]+)\.',  # Title between periods (rough heuristic)
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, ref_text)
            if match:
                potential_title = match.group(1).strip()
                if len(potential_title) > 10 and not potential_title.isdigit():
                    ref.title = potential_title
                    break
        
        # Extract authors (rough heuristic - text before title or year)
        authors_text = ref_text
        if ref.title:
            authors_text = ref_text.split(ref.title)[0]
        elif ref.year:
            parts = ref_text.split(ref.year)
            if len(parts) > 1:
                authors_text = parts[0]
        
        # Parse author names
        ref.authors = self._parse_authors(authors_text)
        
        # Set confidence based on what we extracted
        confidence_factors = []
        if ref.arxiv_id:
            confidence_factors.append(0.4)
        if ref.doi:
            confidence_factors.append(0.3)
        if ref.title:
            confidence_factors.append(0.2)
        if ref.authors:
            confidence_factors.append(0.1)
        
        ref.confidence = sum(confidence_factors)
        
        return ref if ref.confidence > 0.1 else None
    
    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text."""
        for pattern in self.arxiv_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._clean_arxiv_id(match.group(1))
        return None
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and normalize arXiv ID."""
        # Remove version numbers for consistency
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        return arxiv_id
    
    def _parse_authors(self, authors_text: str) -> List[str]:
        """Parse author names from text."""
        if not authors_text:
            return []
        
        # Clean up the text
        authors_text = re.sub(r'^\[\d+\]', '', authors_text).strip()
        authors_text = re.sub(r'^\d+\.', '', authors_text).strip()
        
        # Split on common separators
        separators = [' and ', ', ', ';']
        authors = [authors_text]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(author.split(sep))
            authors = new_authors
        
        # Clean and filter authors
        cleaned_authors = []
        for author in authors:
            author = author.strip(' .,')
            # Filter out non-name text
            if (len(author) > 2 and 
                not author.isdigit() and 
                not re.match(r'^http', author, re.IGNORECASE) and
                not re.match(r'^\d{4}$', author)):  # Not just a year
                cleaned_authors.append(author)
        
        return cleaned_authors[:10]  # Limit to reasonable number
    
    def extract_references(self, extraction_result: Dict[str, Any]) -> List[Reference]:
        """Extract references from any type of extraction result."""
        extraction_method = extraction_result.get('extraction_method', '')
        
        if 'grobid' in extraction_method.lower():
            return self.extract_from_grobid(extraction_result)
        elif 'mistral' in extraction_method.lower():
            return self.extract_from_mistral(extraction_result)
        else:
            # Generic text-based extraction
            content = extraction_result.get('content', '')
            return self._extract_from_text(content)
    
    def resolve_arxiv_references(self, references: List[Reference]) -> List[Reference]:
        """Resolve arXiv IDs in references for better linking."""
        resolved_refs = []
        
        for ref in references:
            # If we already have an arXiv ID, we're good
            if ref.arxiv_id:
                resolved_refs.append(ref)
                continue
            
            # Try to find arXiv ID in title/raw text using fuzzy matching
            # This is a placeholder for more sophisticated resolution
            if ref.title:
                # Could implement fuzzy matching against known arXiv papers
                pass
            
            resolved_refs.append(ref)
        
        return resolved_refs