"""ArXiv API client with rate limiting and response parsing."""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from dateutil.parser import parse as parse_date

from ..utils.name_utils import generate_author_search_queries, NameNormalizer

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for arXiv API (max 3 requests per second)."""

    def __init__(self, max_requests: int = 3, time_window: float = 1.0):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made."""
        async with self._lock:
            now = asyncio.get_event_loop().time()

            # Remove old requests outside the time window
            self.requests = [
                req_time
                for req_time in self.requests
                if now - req_time < self.time_window
            ]

            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                sleep_time = self.time_window - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            # Record this request
            self.requests.append(now)


class ArxivClient:
    """Client for interacting with the arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.rate_limiter = RateLimiter()
        self._own_session = session is None

    async def __aenter__(self):
        if self._own_session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._own_session and self.session:
            await self.session.close()

    async def _ensure_session(self):
        """Ensure we have an aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def _make_request(self, params: Dict[str, Any]) -> str:
        """Make a rate-limited request to the arXiv API."""
        await self.rate_limiter.acquire()
        await self._ensure_session()

        url = f"{self.BASE_URL}?{urlencode(params)}"
        logger.debug(f"Making request to: {url}")

        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv API XML response."""
        try:
            root = ET.fromstring(xml_content)

            # Define namespaces
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }

            papers = []
            entries = root.findall("atom:entry", namespaces)

            for entry in entries:
                paper = self._parse_entry(entry, namespaces)
                if paper:
                    papers.append(paper)

            return papers

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise ValueError(f"Failed to parse arXiv response: {e}")

    def _parse_entry(
        self, entry: ET.Element, namespaces: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse a single entry from arXiv response."""
        try:
            # Basic fields
            title = entry.find("atom:title", namespaces)
            summary = entry.find("atom:summary", namespaces)
            published = entry.find("atom:published", namespaces)
            updated = entry.find("atom:updated", namespaces)

            # arXiv ID from the id field
            id_elem = entry.find("atom:id", namespaces)
            arxiv_id = self._extract_arxiv_id(
                id_elem.text if id_elem is not None else ""
            )

            # Authors
            authors = []
            for author in entry.findall("atom:author", namespaces):
                name = author.find("atom:name", namespaces)
                if name is not None:
                    authors.append(name.text.strip())

            # Categories
            categories = []
            for category in entry.findall("atom:category", namespaces):
                term = category.get("term")
                if term:
                    categories.append(term)

            # Links
            links = entry.findall("atom:link", namespaces)
            url = ""
            pdf_url = ""

            for link in links:
                href = link.get("href", "")
                if link.get("rel") == "alternate":
                    url = href
                elif link.get("title") == "pdf":
                    pdf_url = href

            # arXiv-specific fields
            comment_elem = entry.find("arxiv:comment", namespaces)
            journal_ref_elem = entry.find("arxiv:journal_ref", namespaces)
            doi_elem = entry.find("arxiv:doi", namespaces)

            # Clean and format text
            clean_title = self._clean_text(title.text if title is not None else "")
            clean_abstract = self._clean_text(
                summary.text if summary is not None else ""
            )

            return {
                "id": arxiv_id,
                "title": clean_title,
                "authors": authors,
                "abstract": clean_abstract,
                "published": published.text if published is not None else "",
                "updated": updated.text if updated is not None else "",
                "categories": categories,
                "url": url,
                "pdf_url": pdf_url,
                "comment": (
                    comment_elem.text.strip() if comment_elem is not None else None
                ),
                "journal_ref": (
                    journal_ref_elem.text.strip()
                    if journal_ref_elem is not None
                    else None
                ),
                "doi": doi_elem.text.strip() if doi_elem is not None else None,
            }

        except Exception as e:
            logger.warning(f"Failed to parse entry: {e}")
            return None

    def _extract_arxiv_id(self, id_url: str) -> str:
        """Extract arXiv ID from URL."""
        # arXiv URLs look like: http://arxiv.org/abs/2301.00001v1
        match = re.search(r"arxiv\.org/abs/([^/]+)", id_url)
        return match.group(1) if match else id_url

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and newlines."""
        if not text:
            return ""

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text.strip())

        # Handle common LaTeX symbols
        latex_replacements = {
            r"\n": " ",
            r"\\": "",
            "$": "",
        }

        for pattern, replacement in latex_replacements.items():
            text = text.replace(pattern, replacement)

        return text.strip()

    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        smart_author_search: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for papers on arXiv with smart author name handling."""

        all_papers = []
        seen_ids = set()

        # Generate search query variations (especially for author names)
        if smart_author_search:
            query_variations = generate_author_search_queries(query)
            logger.debug(f"Generated {len(query_variations)} query variations for: {query}")
        else:
            query_variations = [query]

        # Distribute max_results across variations to avoid overwhelming single variations
        results_per_variation = max(1, max_results // len(query_variations))
        remaining_results = max_results

        for i, search_query in enumerate(query_variations):
            if remaining_results <= 0:
                break

            # Build final search query with categories
            final_query = search_query
            if categories:
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                final_query = f"({search_query}) AND ({category_query})"

            # Calculate results for this variation
            current_max = min(results_per_variation, remaining_results, 100)  # arXiv API limit

            params = {
                "search_query": final_query,
                "start": 0,
                "max_results": current_max,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }

            try:
                xml_response = await self._make_request(params)
                variation_papers = self._parse_arxiv_response(xml_response)

                # Filter by date if specified
                if start_date or end_date:
                    variation_papers = self._filter_by_date(variation_papers, start_date, end_date)

                # Add new papers (avoid duplicates)
                new_papers = 0
                for paper in variation_papers:
                    if paper["id"] not in seen_ids:
                        seen_ids.add(paper["id"])
                        all_papers.append(paper)
                        new_papers += 1
                        remaining_results -= 1

                        if remaining_results <= 0:
                            break

                logger.debug(f"Variation {i+1}/{len(query_variations)} '{search_query}': found {new_papers} new papers")

            except Exception as e:
                logger.warning(f"Search variation failed: {search_query}: {e}")
                continue

        # Sort final results by relevance/date
        if sort_by == "relevance":
            # Keep arXiv's relevance ordering, but prefer papers found by earlier (more exact) variations
            pass
        elif sort_by == "lastUpdatedDate":
            all_papers.sort(key=lambda p: p.get("updated", ""), reverse=(sort_order == "descending"))
        elif sort_by == "submittedDate":
            all_papers.sort(key=lambda p: p.get("published", ""), reverse=(sort_order == "descending"))

        return all_papers[:max_results]

    async def get_paper_details(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific paper by arXiv ID."""

        # Clean up the arXiv ID
        arxiv_id = arxiv_id.strip()
        if not arxiv_id:
            raise ValueError("arXiv ID cannot be empty")

        params = {"id_list": arxiv_id, "max_results": 1}

        xml_response = await self._make_request(params)
        papers = self._parse_arxiv_response(xml_response)

        return papers[0] if papers else None

    async def get_recent_papers(
        self, category: str, max_results: int = 10, days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Get recent papers from a specific category."""

        # arXiv API doesn't support wildcard queries, so we search by category only
        # and then filter by date in post-processing
        category_query = f"cat:{category}"
        
        params = {
            "search_query": category_query,
            "start": 0,
            "max_results": min(max_results * 3, 100),  # Get more to filter by date
            "sortBy": "lastUpdatedDate",  # Use supported sort option
            "sortOrder": "descending",
        }

        xml_response = await self._make_request(params)
        papers = self._parse_arxiv_response(xml_response)
        
        # Filter by date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        filtered_papers = self._filter_by_date(
            papers,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Return requested number of results
        return filtered_papers[:max_results]

    def _filter_by_date(
        self,
        papers: List[Dict[str, Any]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter papers by publication date."""
        filtered_papers = []

        for paper in papers:
            try:
                pub_date = parse_date(paper["published"]).date()

                # Check start date
                if start_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
                    if pub_date < start_dt:
                        continue

                # Check end date
                if end_date:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
                    if pub_date > end_dt:
                        continue

                filtered_papers.append(paper)

            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not parse date for paper {paper.get('id', 'unknown')}: {e}"
                )
                # Include papers with unparseable dates to be safe
                filtered_papers.append(paper)

        return filtered_papers

    async def get_author_papers(
        self,
        author_name: str,
        max_results: int = 20,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sort_by: str = "lastUpdatedDate",
        sort_order: str = "descending",
    ) -> List[Dict[str, Any]]:
        """Get all papers by a specific author using smart name matching."""
        
        # Generate author name variations
        name_variations = NameNormalizer.generate_name_variations(author_name)
        logger.debug(f"Searching for author with {len(name_variations)} name variations")
        
        all_papers = []
        seen_ids = set()
        
        # Search with author-specific queries
        for i, name_variant in enumerate(name_variations):
            if len(all_papers) >= max_results:
                break
                
            # Use explicit author search syntax
            author_query = f"au:{name_variant}"
            
            # Add category filter if specified
            if categories:
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                search_query = f"({author_query}) AND ({category_query})"
            else:
                search_query = author_query
            
            # Calculate results for this variation
            current_max = min(max_results - len(all_papers), 50)  # Get up to 50 per variation
            
            params = {
                "search_query": search_query,
                "start": 0,
                "max_results": current_max,
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
            
            try:
                xml_response = await self._make_request(params)
                variation_papers = self._parse_arxiv_response(xml_response)
                
                # Filter by date if specified
                if start_date or end_date:
                    variation_papers = self._filter_by_date(variation_papers, start_date, end_date)
                
                # Add new papers (avoid duplicates)
                new_papers = 0
                for paper in variation_papers:
                    if paper["id"] not in seen_ids:
                        # Verify author name appears in paper authors
                        if self._author_matches_paper(author_name, paper["authors"]):
                            seen_ids.add(paper["id"])
                            all_papers.append(paper)
                            new_papers += 1
                
                logger.debug(f"Author variation {i+1}/{len(name_variations)} '{name_variant}': found {new_papers} new papers")
                
            except Exception as e:
                logger.warning(f"Author search variation failed: {name_variant}: {e}")
                continue
        
        return all_papers[:max_results]
    
    def _author_matches_paper(self, search_name: str, paper_authors: List[str]) -> bool:
        """Check if the search name matches any author in the paper."""
        search_variations = set(NameNormalizer.generate_name_variations(search_name))
        
        for paper_author in paper_authors:
            paper_variations = set(NameNormalizer.generate_name_variations(paper_author))
            
            # Check for any overlap between search and paper author variations
            if search_variations & paper_variations:
                return True
        
        return False

    async def find_similar_papers(
        self,
        reference_paper_id: str,
        max_results: int = 10,
        similarity_method: str = "keywords",
    ) -> List[Dict[str, Any]]:
        """Find papers similar to a reference paper."""
        
        # Get the reference paper
        reference_paper = await self.get_paper_details(reference_paper_id)
        if not reference_paper:
            raise ValueError(f"Reference paper {reference_paper_id} not found")
        
        similar_papers = []
        seen_ids = {reference_paper_id}
        
        if similarity_method == "keywords":
            # Extract keywords from title and abstract
            title_words = self._extract_keywords(reference_paper.get("title", ""))
            abstract_words = self._extract_keywords(reference_paper.get("abstract", ""))
            
            # Combine and weight keywords (title words get higher priority)
            all_keywords = list(set(title_words[:5] + abstract_words[:10]))
            
            # Search using combinations of keywords
            for i in range(min(3, len(all_keywords))):
                if len(similar_papers) >= max_results:
                    break
                    
                # Use subsets of keywords
                if i == 0 and len(all_keywords) >= 3:
                    query_keywords = all_keywords[:3]
                elif i == 1 and len(all_keywords) >= 2:
                    query_keywords = all_keywords[:2]
                else:
                    query_keywords = all_keywords[:1]
                
                search_query = " AND ".join(query_keywords)
                
                try:
                    params = {
                        "search_query": search_query,
                        "start": 0,
                        "max_results": min(20, max_results * 2),
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    }
                    
                    xml_response = await self._make_request(params)
                    papers = self._parse_arxiv_response(xml_response)
                    
                    for paper in papers:
                        if (paper["id"] not in seen_ids and 
                            len(similar_papers) < max_results):
                            seen_ids.add(paper["id"])
                            similar_papers.append(paper)
                            
                except Exception as e:
                    logger.warning(f"Similar paper search failed for keywords {query_keywords}: {e}")
                    continue
        
        elif similarity_method == "categories":
            # Find papers in same categories
            categories = reference_paper.get("categories", [])
            if categories:
                category_query = " OR ".join([f"cat:{cat}" for cat in categories[:3]])
                
                try:
                    params = {
                        "search_query": category_query,
                        "start": 0,
                        "max_results": min(50, max_results * 3),
                        "sortBy": "relevance",
                        "sortOrder": "descending",
                    }
                    
                    xml_response = await self._make_request(params)
                    papers = self._parse_arxiv_response(xml_response)
                    
                    for paper in papers[:max_results]:
                        if paper["id"] not in seen_ids:
                            similar_papers.append(paper)
                            
                except Exception as e:
                    logger.warning(f"Category-based similarity search failed: {e}")
        
        elif similarity_method == "authors":
            # Find papers by same authors
            authors = reference_paper.get("authors", [])
            for author in authors[:3]:  # Check first 3 authors
                if len(similar_papers) >= max_results:
                    break
                    
                try:
                    author_papers = await self.get_author_papers(
                        author, 
                        max_results=10,
                        sort_by="lastUpdatedDate"
                    )
                    
                    for paper in author_papers:
                        if (paper["id"] not in seen_ids and 
                            len(similar_papers) < max_results):
                            seen_ids.add(paper["id"])
                            similar_papers.append(paper)
                            
                except Exception as e:
                    logger.warning(f"Author-based similarity search failed for {author}: {e}")
                    continue
        
        return similar_papers
    
    def _extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return []
        
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'shall', 'we', 'they', 'i', 'you', 'he', 'she',
            'it', 'our', 'your', 'their', 'his', 'her', 'its', 'paper', 'study', 'research',
            'show', 'shows', 'present', 'presented', 'propose', 'proposed', 'using', 'used'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
        
        # Filter stop words and get unique keywords
        keywords = []
        seen = set()
        for word in words:
            if word not in stop_words and word not in seen:
                keywords.append(word)
                seen.add(word)
        
        return keywords[:15]  # Return top 15 keywords
