"""Citation formatting utilities for ArXiv papers."""

import re
from datetime import datetime
from typing import Dict, Any, List
from urllib.parse import urlparse


class CitationFormatter:
    """Formats ArXiv papers into various citation styles."""
    
    @classmethod
    def _clean_title(cls, title: str) -> str:
        """Clean title for citation formatting."""
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title.strip())
        
        # Ensure title ends with period if it doesn't already have punctuation
        if title and not title[-1] in '.!?':
            title += '.'
        
        return title
    
    @classmethod
    def _format_authors(cls, authors: List[str], style: str = "apa") -> str:
        """Format author list according to citation style."""
        if not authors:
            return ""
        
        if style == "bibtex":
            # BibTeX prefers "Last, First and Last, First"
            formatted_authors = []
            for author in authors:
                if ',' in author:
                    # Already in "Last, First" format
                    formatted_authors.append(author)
                else:
                    # Convert "First Last" to "Last, First"
                    parts = author.strip().split()
                    if len(parts) >= 2:
                        last = parts[-1]
                        first = ' '.join(parts[:-1])
                        formatted_authors.append(f"{last}, {first}")
                    else:
                        formatted_authors.append(author)
            return ' and '.join(formatted_authors)
        
        elif style == "apa":
            # APA: "Last, F. M." format
            formatted_authors = []
            for author in authors[:20]:  # APA limits to 20 authors
                if ',' in author:
                    # Already in "Last, First" format
                    parts = author.split(',', 1)
                    last = parts[0].strip()
                    first = parts[1].strip()
                else:
                    # Convert "First Last" to "Last, First"
                    name_parts = author.strip().split()
                    if len(name_parts) >= 2:
                        last = name_parts[-1]
                        first = ' '.join(name_parts[:-1])
                    else:
                        last = author
                        first = ""
                
                # Convert first names to initials
                if first:
                    initials = '. '.join([name[0].upper() for name in first.split() if name]) + '.'
                    formatted_authors.append(f"{last}, {initials}")
                else:
                    formatted_authors.append(last)
            
            if len(authors) == 1:
                return formatted_authors[0]
            elif len(authors) == 2:
                return f"{formatted_authors[0]}, & {formatted_authors[1]}"
            elif len(authors) <= 20:
                return ', '.join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
            else:
                return ', '.join(formatted_authors[:19]) + ", ... " + formatted_authors[-1]
        
        elif style == "mla":
            # MLA: "Last, First M." for first author, "First M. Last" for others
            formatted_authors = []
            for i, author in enumerate(authors):
                if ',' in author:
                    parts = author.split(',', 1)
                    last = parts[0].strip()
                    first = parts[1].strip()
                else:
                    name_parts = author.strip().split()
                    if len(name_parts) >= 2:
                        last = name_parts[-1]
                        first = ' '.join(name_parts[:-1])
                    else:
                        last = author
                        first = ""
                
                if i == 0:
                    # First author: "Last, First"
                    formatted_authors.append(f"{last}, {first}" if first else last)
                else:
                    # Other authors: "First Last"
                    formatted_authors.append(f"{first} {last}" if first else last)
            
            if len(authors) == 1:
                return formatted_authors[0]
            elif len(authors) == 2:
                return f"{formatted_authors[0]}, and {formatted_authors[1]}"
            elif len(authors) <= 3:
                return ', '.join(formatted_authors[:-1]) + f", and {formatted_authors[-1]}"
            else:
                return f"{formatted_authors[0]}, et al."
        
        elif style == "chicago":
            # Chicago: Similar to MLA but with different punctuation
            formatted_authors = []
            for i, author in enumerate(authors):
                if ',' in author:
                    parts = author.split(',', 1)
                    last = parts[0].strip()
                    first = parts[1].strip()
                else:
                    name_parts = author.strip().split()
                    if len(name_parts) >= 2:
                        last = name_parts[-1]
                        first = ' '.join(name_parts[:-1])
                    else:
                        last = author
                        first = ""
                
                if i == 0:
                    formatted_authors.append(f"{last}, {first}" if first else last)
                else:
                    formatted_authors.append(f"{first} {last}" if first else last)
            
            if len(authors) == 1:
                return formatted_authors[0]
            elif len(authors) <= 3:
                return ', and '.join(formatted_authors)
            else:
                return f"{formatted_authors[0]} et al."
        
        return ', '.join(authors)
    
    @classmethod
    def _extract_year(cls, date_str: str) -> str:
        """Extract year from date string."""
        if not date_str:
            return ""
        
        # Try to parse various date formats
        try:
            # ArXiv typically uses ISO format
            if 'T' in date_str:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return str(date_obj.year)
        except:
            # Fallback: extract 4-digit year with regex
            year_match = re.search(r'\b(20\d{2}|19\d{2})\b', date_str)
            return year_match.group(1) if year_match else ""
    
    @classmethod
    def _clean_arxiv_id(cls, arxiv_id: str) -> str:
        """Clean ArXiv ID for citation."""
        # Remove version numbers (e.g., v1, v2)
        return re.sub(r'v\d+$', '', arxiv_id)
    
    @classmethod
    def format_bibtex(cls, paper: Dict[str, Any]) -> str:
        """Format paper as BibTeX entry."""
        arxiv_id = cls._clean_arxiv_id(paper.get('id', ''))
        title = cls._clean_title(paper.get('title', ''))
        authors = cls._format_authors(paper.get('authors', []), 'bibtex')
        year = cls._extract_year(paper.get('published', ''))
        abstract = paper.get('abstract', '')
        categories = ', '.join(paper.get('categories', []))
        
        # Create BibTeX key (first author last name + year + first word of title)
        key_parts = []
        if authors:
            first_author = authors.split(' and ')[0]
            if ',' in first_author:
                last_name = first_author.split(',')[0].strip()
                last_name = re.sub(r'[^a-zA-Z]', '', last_name)
                key_parts.append(last_name)
        
        if year:
            key_parts.append(year)
        
        if title:
            first_word = re.findall(r'\b[a-zA-Z]+\b', title)
            if first_word:
                key_parts.append(first_word[0])
        
        bibtex_key = ''.join(key_parts) or arxiv_id.replace('/', '').replace('.', '')
        
        bibtex = f"@misc{{{bibtex_key},\n"
        bibtex += f"  title={{{title}}},\n"
        if authors:
            bibtex += f"  author={{{authors}}},\n"
        if year:
            bibtex += f"  year={{{year}}},\n"
        bibtex += f"  eprint={{{arxiv_id}}},\n"
        bibtex += f"  archivePrefix={{arXiv}},\n"
        if categories:
            primary_cat = categories.split(',')[0].strip()
            bibtex += f"  primaryClass={{{primary_cat}}},\n"
        if paper.get('url'):
            bibtex += f"  url={{{paper['url']}}},\n"
        if abstract:
            # Clean abstract for BibTeX
            clean_abstract = re.sub(r'\s+', ' ', abstract.strip())
            bibtex += f"  abstract={{{clean_abstract}}},\n"
        bibtex += "}"
        
        return bibtex
    
    @classmethod
    def format_apa(cls, paper: Dict[str, Any]) -> str:
        """Format paper as APA citation."""
        authors = cls._format_authors(paper.get('authors', []), 'apa')
        year = cls._extract_year(paper.get('published', ''))
        title = cls._clean_title(paper.get('title', ''))
        arxiv_id = cls._clean_arxiv_id(paper.get('id', ''))
        
        citation = ""
        if authors:
            citation += f"{authors} "
        if year:
            citation += f"({year}). "
        if title:
            citation += f"{title} "
        citation += f"*arXiv preprint arXiv:{arxiv_id}*."
        
        return citation.strip()
    
    @classmethod
    def format_mla(cls, paper: Dict[str, Any]) -> str:
        """Format paper as MLA citation."""
        authors = cls._format_authors(paper.get('authors', []), 'mla')
        title = paper.get('title', '').strip()
        if title and not title.endswith('.'):
            title += '.'
        arxiv_id = cls._clean_arxiv_id(paper.get('id', ''))
        year = cls._extract_year(paper.get('published', ''))
        
        citation = ""
        if authors:
            citation += f"{authors} "
        if title:
            citation += f'"{title}" '
        citation += f"*arXiv*, "
        if year:
            citation += f"{year}, "
        citation += f"arXiv:{arxiv_id}."
        
        return citation.strip()
    
    @classmethod
    def format_chicago(cls, paper: Dict[str, Any]) -> str:
        """Format paper as Chicago citation."""
        authors = cls._format_authors(paper.get('authors', []), 'chicago')
        title = paper.get('title', '').strip()
        if title and not title.endswith('.'):
            title += '.'
        arxiv_id = cls._clean_arxiv_id(paper.get('id', ''))
        year = cls._extract_year(paper.get('published', ''))
        
        citation = ""
        if authors:
            citation += f"{authors}. "
        if title:
            citation += f'"{title}" '
        citation += f"arXiv preprint arXiv:{arxiv_id} "
        if year:
            citation += f"({year})"
        citation += "."
        
        return citation.strip()
    
    @classmethod
    def format_citation(cls, paper: Dict[str, Any], style: str = "apa") -> str:
        """Format paper citation in specified style."""
        style = style.lower()
        
        if style == "bibtex":
            return cls.format_bibtex(paper)
        elif style == "apa":
            return cls.format_apa(paper)
        elif style == "mla":
            return cls.format_mla(paper)
        elif style == "chicago":
            return cls.format_chicago(paper)
        else:
            raise ValueError(f"Unsupported citation style: {style}")


def format_bibliography(papers: List[Dict[str, Any]], style: str = "apa") -> str:
    """Format multiple papers as a bibliography."""
    if not papers:
        return "No papers to format."
    
    citations = []
    for paper in papers:
        try:
            citation = CitationFormatter.format_citation(paper, style)
            citations.append(citation)
        except Exception as e:
            # Skip papers that can't be formatted
            continue
    
    if style.lower() == "bibtex":
        return "\n\n".join(citations)
    else:
        # For other styles, add hanging indent and numbering
        formatted_citations = []
        for i, citation in enumerate(citations, 1):
            formatted_citations.append(f"{i}. {citation}")
        
        return "\n\n".join(formatted_citations)