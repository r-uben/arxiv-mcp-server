"""Semantic Scholar API client for citation discovery and paper lookup."""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import os

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarPaper:
    """Semantic Scholar paper information."""
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: List[Dict[str, Any]]
    year: Optional[int]
    citation_count: int
    reference_count: int
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    venue: Optional[str] = None


class SemanticScholarClient:
    """Client for Semantic Scholar Academic Graph API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key for higher rate limits."""
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        # Rate limiting (1 req/sec with API key, conservative without)
        self.rate_limit_delay = 1.0 if self.api_key else 2.0  # seconds between requests
        self.last_request_time = 0.0
        
        # Session will be created when needed
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"Semantic Scholar client initialized {'with' if self.api_key else 'without'} API key")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if not self.session:
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            self.session = aiohttp.ClientSession(headers=headers)
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make rate-limited request to Semantic Scholar API."""
        await self._ensure_session()
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    # Rate limited - wait longer and retry once
                    logger.warning("Rate limited by Semantic Scholar API, waiting...")
                    await asyncio.sleep(self.rate_limit_delay * 2)
                    async with self.session.get(url, params=params) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.json()
                        else:
                            raise aiohttp.ClientError(f"API request failed after retry: {retry_response.status}")
                elif response.status == 404:
                    return None  # Paper not found
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"API request failed: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"Semantic Scholar API request failed: {e}")
            raise
    
    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[SemanticScholarPaper]:
        """Get paper information by arXiv ID."""
        try:
            # Clean arXiv ID
            clean_id = arxiv_id.replace('arXiv:', '').replace('v1', '').replace('v2', '').replace('v3', '')
            
            # Semantic Scholar expects format like "arXiv:2301.07041"
            search_id = f"arXiv:{clean_id}"
            
            result = await self._make_request(f"paper/{search_id}", {
                'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
            })
            
            if result:
                return self._parse_paper_result(result)
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper by arXiv ID {arxiv_id}: {e}")
            return None
    
    async def get_paper_by_doi(self, doi: str) -> Optional[SemanticScholarPaper]:
        """Get paper information by DOI."""
        try:
            # Clean DOI
            if not doi.startswith('10.'):
                return None
            
            result = await self._make_request(f"paper/DOI:{doi}", {
                'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
            })
            
            if result:
                return self._parse_paper_result(result)
            return None
            
        except Exception as e:
            logger.error(f"Error getting paper by DOI {doi}: {e}")
            return None
    
    async def search_paper_by_title(self, title: str, limit: int = 5) -> List[SemanticScholarPaper]:
        """Search for papers by title."""
        try:
            params = {
                'query': title,
                'limit': limit,
                'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
            }
            
            result = await self._make_request("paper/search", params)
            
            if result and 'data' in result:
                papers = []
                for paper_data in result['data']:
                    paper = self._parse_paper_result(paper_data)
                    if paper:
                        papers.append(paper)
                return papers
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching paper by title '{title}': {e}")
            return []
    
    async def get_citations(self, paper_id: str, limit: int = 50) -> List[SemanticScholarPaper]:
        """Get papers that cite the given paper."""
        try:
            params = {
                'limit': limit,
                'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
            }
            
            result = await self._make_request(f"paper/{paper_id}/citations", params)
            
            if result and 'data' in result:
                citing_papers = []
                for citation_data in result['data']:
                    # The citing paper is in the 'citingPaper' field
                    paper_data = citation_data.get('citingPaper')
                    if paper_data:
                        paper = self._parse_paper_result(paper_data)
                        if paper:
                            citing_papers.append(paper)
                
                return citing_papers
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting citations for paper {paper_id}: {e}")
            return []
    
    async def get_references(self, paper_id: str, limit: int = 50) -> List[SemanticScholarPaper]:
        """Get papers referenced by the given paper."""
        try:
            params = {
                'limit': limit,
                'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
            }
            
            result = await self._make_request(f"paper/{paper_id}/references", params)
            
            if result and 'data' in result:
                referenced_papers = []
                for reference_data in result['data']:
                    # The referenced paper is in the 'citedPaper' field
                    paper_data = reference_data.get('citedPaper')
                    if paper_data:
                        paper = self._parse_paper_result(paper_data)
                        if paper:
                            referenced_papers.append(paper)
                
                return referenced_papers
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting references for paper {paper_id}: {e}")
            return []
    
    def _parse_paper_result(self, paper_data: Dict[str, Any]) -> Optional[SemanticScholarPaper]:
        """Parse Semantic Scholar paper data into SemanticScholarPaper object."""
        try:
            # Extract external IDs
            external_ids = paper_data.get('externalIds', {}) or {}
            arxiv_id = external_ids.get('ArXiv')
            doi = external_ids.get('DOI')
            
            # Parse authors
            authors = []
            for author_data in paper_data.get('authors', []):
                authors.append({
                    'name': author_data.get('name', ''),
                    'author_id': author_data.get('authorId')
                })
            
            # Extract venue
            venue_data = paper_data.get('venue')
            venue = venue_data if isinstance(venue_data, str) else None
            
            return SemanticScholarPaper(
                paper_id=paper_data.get('paperId', ''),
                title=paper_data.get('title', ''),
                abstract=paper_data.get('abstract'),
                authors=authors,
                year=paper_data.get('year'),
                citation_count=paper_data.get('citationCount', 0),
                reference_count=paper_data.get('referenceCount', 0),
                arxiv_id=arxiv_id,
                doi=doi,
                url=paper_data.get('url'),
                venue=venue
            )
            
        except Exception as e:
            logger.error(f"Error parsing paper result: {e}")
            return None
    
    async def resolve_paper_by_reference(self, title: str, authors: List[str] = None, year: str = None) -> Optional[SemanticScholarPaper]:
        """Try to resolve a paper by title and optional metadata."""
        try:
            # First try exact title search
            papers = await self.search_paper_by_title(title, limit=10)
            
            if not papers:
                return None
            
            # If we have additional metadata, try to find best match
            if year or authors:
                best_match = None
                best_score = 0
                
                for paper in papers:
                    score = 0
                    
                    # Year matching
                    if year and paper.year:
                        if str(paper.year) == str(year):
                            score += 0.5
                        elif abs(paper.year - int(year)) <= 1:  # Allow 1 year difference
                            score += 0.3
                    
                    # Author matching (simplified)
                    if authors and paper.authors:
                        paper_author_names = [a['name'].lower() for a in paper.authors]
                        for ref_author in authors:
                            ref_author_lower = ref_author.lower()
                            for paper_author in paper_author_names:
                                # Simple substring match
                                if (ref_author_lower in paper_author or 
                                    paper_author in ref_author_lower):
                                    score += 0.1
                                    break
                    
                    if score > best_score:
                        best_score = score
                        best_match = paper
                
                # Return best match if score is reasonable
                if best_match and best_score > 0.3:
                    return best_match
            
            # Otherwise return first result (highest relevance)
            return papers[0]
            
        except Exception as e:
            logger.error(f"Error resolving paper by reference: {e}")
            return None
    
    async def get_paper_batch(self, paper_ids: List[str], max_batch_size: int = 500) -> List[SemanticScholarPaper]:
        """Get multiple papers in batch (more efficient for large numbers)."""
        papers = []
        
        # Process in batches to respect API limits
        for i in range(0, len(paper_ids), max_batch_size):
            batch_ids = paper_ids[i:i + max_batch_size]
            
            try:
                # Batch endpoint - comma-separated IDs
                ids_str = ','.join(batch_ids)
                params = {
                    'fields': 'paperId,title,abstract,authors,year,citationCount,referenceCount,externalIds,venue,url'
                }
                
                result = await self._make_request(f"paper/{ids_str}", params)
                
                if isinstance(result, list):
                    # Batch response is a list
                    for paper_data in result:
                        if paper_data:  # Skip null entries
                            paper = self._parse_paper_result(paper_data)
                            if paper:
                                papers.append(paper)
                elif isinstance(result, dict):
                    # Single paper response
                    paper = self._parse_paper_result(result)
                    if paper:
                        papers.append(paper)
                
                # Small delay between batches
                if i + max_batch_size < len(paper_ids):
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in batch paper request: {e}")
                continue
        
        return papers
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None