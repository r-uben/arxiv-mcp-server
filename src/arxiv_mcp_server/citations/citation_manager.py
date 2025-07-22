"""Citation management system combining reference extraction and Semantic Scholar lookup."""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .reference_extractor import ReferenceExtractor, Reference
from .semantic_scholar import SemanticScholarClient, SemanticScholarPaper
from ..storage.paper_library import PaperLibrary

logger = logging.getLogger(__name__)


@dataclass
class CitationLink:
    """Represents a citation relationship between papers."""
    citing_paper: str  # arXiv ID of paper making the citation
    cited_paper: str   # arXiv ID of paper being cited
    citation_text: str  # Raw citation text
    confidence: float   # 0-1 confidence in the link
    resolved_paper: Optional[SemanticScholarPaper] = None


@dataclass
class CitationNetwork:
    """Network of citations around a paper."""
    center_paper: str  # arXiv ID of the center paper
    references: List[CitationLink]  # Papers this paper cites
    citations: List[CitationLink]   # Papers that cite this paper
    depth: int = 1  # Network depth (1 = direct connections only)
    total_papers: int = 0


class CitationManager:
    """Manages citation extraction, resolution, and network building."""
    
    def __init__(self, library: Optional[PaperLibrary] = None, semantic_scholar_api_key: Optional[str] = None):
        """Initialize citation manager."""
        self.ref_extractor = ReferenceExtractor()
        self.semantic_client = SemanticScholarClient(api_key=semantic_scholar_api_key)
        self.library = library
        
        # Cache for resolved papers to avoid repeated API calls
        self._paper_cache: Dict[str, SemanticScholarPaper] = {}
        self._citation_cache: Dict[str, List[SemanticScholarPaper]] = {}
        
        logger.info("Citation manager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.semantic_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.semantic_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def extract_and_resolve_references(
        self, 
        arxiv_id: str, 
        extraction_result: Dict[str, Any]
    ) -> List[CitationLink]:
        """Extract references from paper and resolve them to arXiv papers."""
        try:
            # Extract references from extraction result
            references = self.ref_extractor.extract_references(extraction_result)
            logger.info(f"Extracted {len(references)} references from {arxiv_id}")
            
            # Resolve references to papers
            citation_links = []
            for ref in references:
                link = await self._resolve_reference_to_citation(arxiv_id, ref)
                if link:
                    citation_links.append(link)
            
            # Store in library if available
            if self.library:
                await self._store_citations_in_library(citation_links)
            
            logger.info(f"Resolved {len(citation_links)} references to citations for {arxiv_id}")
            return citation_links
            
        except Exception as e:
            logger.error(f"Error extracting and resolving references for {arxiv_id}: {e}")
            return []
    
    async def _resolve_reference_to_citation(
        self, 
        citing_paper: str, 
        reference: Reference
    ) -> Optional[CitationLink]:
        """Resolve a single reference to a citation link."""
        try:
            resolved_paper = None
            
            # Try different resolution strategies in order of reliability
            
            # 1. Direct arXiv ID lookup
            if reference.arxiv_id:
                resolved_paper = await self._get_paper_by_arxiv_id(reference.arxiv_id)
                if resolved_paper:
                    return CitationLink(
                        citing_paper=citing_paper,
                        cited_paper=reference.arxiv_id,
                        citation_text=reference.raw_text,
                        confidence=0.9,
                        resolved_paper=resolved_paper
                    )
            
            # 2. DOI lookup
            if reference.doi and not resolved_paper:
                resolved_paper = await self._get_paper_by_doi(reference.doi)
                if resolved_paper and resolved_paper.arxiv_id:
                    return CitationLink(
                        citing_paper=citing_paper,
                        cited_paper=resolved_paper.arxiv_id,
                        citation_text=reference.raw_text,
                        confidence=0.8,
                        resolved_paper=resolved_paper
                    )
            
            # 3. Title + metadata search
            if reference.title and not resolved_paper:
                resolved_paper = await self.semantic_client.resolve_paper_by_reference(
                    title=reference.title,
                    authors=reference.authors,
                    year=reference.year
                )
                if resolved_paper and resolved_paper.arxiv_id:
                    return CitationLink(
                        citing_paper=citing_paper,
                        cited_paper=resolved_paper.arxiv_id,
                        citation_text=reference.raw_text,
                        confidence=0.6,
                        resolved_paper=resolved_paper
                    )
            
            # If we found a paper but no arXiv ID, still create a link for the reference
            if resolved_paper:
                return CitationLink(
                    citing_paper=citing_paper,
                    cited_paper=f"s2:{resolved_paper.paper_id}",  # Use Semantic Scholar ID
                    citation_text=reference.raw_text,
                    confidence=0.4,
                    resolved_paper=resolved_paper
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error resolving reference: {e}")
            return None
    
    async def _get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[SemanticScholarPaper]:
        """Get paper by arXiv ID with caching."""
        if arxiv_id in self._paper_cache:
            return self._paper_cache[arxiv_id]
        
        paper = await self.semantic_client.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            self._paper_cache[arxiv_id] = paper
        
        return paper
    
    async def _get_paper_by_doi(self, doi: str) -> Optional[SemanticScholarPaper]:
        """Get paper by DOI with caching."""
        cache_key = f"doi:{doi}"
        if cache_key in self._paper_cache:
            return self._paper_cache[cache_key]
        
        paper = await self.semantic_client.get_paper_by_doi(doi)
        if paper:
            self._paper_cache[cache_key] = paper
        
        return paper
    
    async def find_citing_papers(self, arxiv_id: str, limit: int = 50) -> List[SemanticScholarPaper]:
        """Find papers that cite the given arXiv paper."""
        try:
            # Check cache first
            cache_key = f"citing:{arxiv_id}"
            if cache_key in self._citation_cache:
                return self._citation_cache[cache_key]
            
            # Get paper info first
            paper = await self._get_paper_by_arxiv_id(arxiv_id)
            if not paper:
                logger.warning(f"Paper {arxiv_id} not found in Semantic Scholar")
                return []
            
            # Get citing papers
            citing_papers = await self.semantic_client.get_citations(paper.paper_id, limit)
            
            # Cache the results
            self._citation_cache[cache_key] = citing_papers
            
            logger.info(f"Found {len(citing_papers)} papers citing {arxiv_id}")
            return citing_papers
            
        except Exception as e:
            logger.error(f"Error finding citing papers for {arxiv_id}: {e}")
            return []
    
    async def find_referenced_papers(self, arxiv_id: str, limit: int = 50) -> List[SemanticScholarPaper]:
        """Find papers referenced by the given arXiv paper."""
        try:
            # Check cache first
            cache_key = f"references:{arxiv_id}"
            if cache_key in self._citation_cache:
                return self._citation_cache[cache_key]
            
            # Get paper info first
            paper = await self._get_paper_by_arxiv_id(arxiv_id)
            if not paper:
                logger.warning(f"Paper {arxiv_id} not found in Semantic Scholar")
                return []
            
            # Get referenced papers
            referenced_papers = await self.semantic_client.get_references(paper.paper_id, limit)
            
            # Cache the results
            self._citation_cache[cache_key] = referenced_papers
            
            logger.info(f"Found {len(referenced_papers)} papers referenced by {arxiv_id}")
            return referenced_papers
            
        except Exception as e:
            logger.error(f"Error finding referenced papers for {arxiv_id}: {e}")
            return []
    
    async def build_citation_network(
        self, 
        center_arxiv_id: str, 
        depth: int = 1,
        max_papers_per_level: int = 20
    ) -> CitationNetwork:
        """Build a citation network around a paper."""
        try:
            logger.info(f"Building citation network for {center_arxiv_id}, depth={depth}")
            
            network = CitationNetwork(
                center_paper=center_arxiv_id,
                references=[],
                citations=[],
                depth=depth
            )
            
            processed_papers: Set[str] = set()
            papers_to_process = [center_arxiv_id]
            
            for current_depth in range(depth):
                next_papers = []
                
                for paper_id in papers_to_process:
                    if paper_id in processed_papers:
                        continue
                    
                    processed_papers.add(paper_id)
                    
                    # Get papers this one references
                    if current_depth == 0 or len(network.references) < max_papers_per_level:
                        referenced = await self.find_referenced_papers(paper_id, max_papers_per_level)
                        for ref_paper in referenced:
                            if ref_paper.arxiv_id:
                                link = CitationLink(
                                    citing_paper=paper_id,
                                    cited_paper=ref_paper.arxiv_id,
                                    citation_text=f"Referenced by {paper_id}",
                                    confidence=0.9,
                                    resolved_paper=ref_paper
                                )
                                network.references.append(link)
                                
                                if current_depth + 1 < depth:
                                    next_papers.append(ref_paper.arxiv_id)
                    
                    # Get papers that cite this one
                    if current_depth == 0 or len(network.citations) < max_papers_per_level:
                        citing = await self.find_citing_papers(paper_id, max_papers_per_level)
                        for citing_paper in citing:
                            if citing_paper.arxiv_id:
                                link = CitationLink(
                                    citing_paper=citing_paper.arxiv_id,
                                    cited_paper=paper_id,
                                    citation_text=f"Cites {paper_id}",
                                    confidence=0.9,
                                    resolved_paper=citing_paper
                                )
                                network.citations.append(link)
                                
                                if current_depth + 1 < depth:
                                    next_papers.append(citing_paper.arxiv_id)
                
                papers_to_process = list(set(next_papers))  # Remove duplicates
            
            # Calculate total unique papers in network
            all_papers = set([center_arxiv_id])
            for link in network.references + network.citations:
                all_papers.add(link.citing_paper)
                all_papers.add(link.cited_paper)
            
            network.total_papers = len(all_papers)
            
            logger.info(f"Built citation network: {len(network.references)} references, "
                       f"{len(network.citations)} citations, {network.total_papers} total papers")
            
            return network
            
        except Exception as e:
            logger.error(f"Error building citation network for {center_arxiv_id}: {e}")
            return CitationNetwork(center_paper=center_arxiv_id, references=[], citations=[])
    
    async def _store_citations_in_library(self, citation_links: List[CitationLink]):
        """Store citation links in the library database."""
        if not self.library:
            return
        
        try:
            import sqlite3
            from datetime import datetime
            
            conn = sqlite3.connect(self.library.db_path)
            cursor = conn.cursor()
            
            for link in citation_links:
                cursor.execute('''
                    INSERT OR REPLACE INTO citations 
                    (citing_paper, cited_paper, citation_text, resolved_arxiv_id, added_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    link.citing_paper,
                    link.cited_paper if not link.cited_paper.startswith('s2:') else None,
                    link.citation_text,
                    link.cited_paper if link.cited_paper.startswith('20') else None,  # Only arXiv IDs
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Stored {len(citation_links)} citation links in library")
            
        except Exception as e:
            logger.error(f"Error storing citations in library: {e}")
    
    async def get_paper_recommendations(
        self, 
        arxiv_id: str, 
        recommendation_type: str = "citing",  # "citing", "references", "related"
        limit: int = 10
    ) -> List[SemanticScholarPaper]:
        """Get paper recommendations based on citation patterns."""
        try:
            if recommendation_type == "citing":
                # Papers that cite this paper (likely interested readers)
                return await self.find_citing_papers(arxiv_id, limit)
            
            elif recommendation_type == "references":
                # Papers referenced by this paper (background/foundation)
                return await self.find_referenced_papers(arxiv_id, limit)
            
            elif recommendation_type == "related":
                # Papers that share citations (co-cited papers)
                network = await self.build_citation_network(arxiv_id, depth=1, max_papers_per_level=50)
                
                # Extract unique papers from the network, excluding the center paper
                related_papers = []
                seen_ids = {arxiv_id}
                
                for link in network.references + network.citations:
                    if link.resolved_paper:
                        paper_id = link.resolved_paper.arxiv_id or link.resolved_paper.paper_id
                        if paper_id not in seen_ids:
                            related_papers.append(link.resolved_paper)
                            seen_ids.add(paper_id)
                
                # Sort by citation count (impact) and return top results
                related_papers.sort(key=lambda p: p.citation_count, reverse=True)
                return related_papers[:limit]
            
            else:
                logger.warning(f"Unknown recommendation type: {recommendation_type}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting paper recommendations: {e}")
            return []
    
    def export_citation_network(self, network: CitationNetwork, format_type: str = "json") -> str:
        """Export citation network in various formats."""
        try:
            if format_type == "json":
                # Convert to JSON-serializable format
                network_dict = {
                    "center_paper": network.center_paper,
                    "depth": network.depth,
                    "total_papers": network.total_papers,
                    "references": [
                        {
                            "citing_paper": link.citing_paper,
                            "cited_paper": link.cited_paper,
                            "citation_text": link.citation_text,
                            "confidence": link.confidence,
                            "resolved_paper": asdict(link.resolved_paper) if link.resolved_paper else None
                        }
                        for link in network.references
                    ],
                    "citations": [
                        {
                            "citing_paper": link.citing_paper,
                            "cited_paper": link.cited_paper,
                            "citation_text": link.citation_text,
                            "confidence": link.confidence,
                            "resolved_paper": asdict(link.resolved_paper) if link.resolved_paper else None
                        }
                        for link in network.citations
                    ]
                }
                
                return json.dumps(network_dict, indent=2)
            
            elif format_type == "graphml":
                # Basic GraphML format for network visualization
                nodes = set([network.center_paper])
                edges = []
                
                for link in network.references + network.citations:
                    nodes.add(link.citing_paper)
                    nodes.add(link.cited_paper)
                    edges.append((link.citing_paper, link.cited_paper, link.citation_text))
                
                graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
                graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
                graphml.append('<graph id="citation_network" edgedefault="directed">')
                
                # Nodes
                for node in nodes:
                    graphml.append(f'  <node id="{node}"/>')
                
                # Edges
                for i, (source, target, label) in enumerate(edges):
                    graphml.append(f'  <edge id="e{i}" source="{source}" target="{target}"/>')
                
                graphml.append('</graph>')
                graphml.append('</graphml>')
                
                return '\n'.join(graphml)
            
            else:
                logger.warning(f"Unknown export format: {format_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Error exporting citation network: {e}")
            return ""