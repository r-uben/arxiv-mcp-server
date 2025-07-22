"""Storage manager for integrating paper library with existing arXiv system."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import asyncio

from .paper_library import PaperLibrary, Paper, Collection
from ..api.arxiv_client import ArxivClient
from ..extraction.smart_extractor import SmartPDFExtractor

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages paper storage, integrating library with arXiv and extraction systems."""
    
    def __init__(self, library_path: Optional[Path] = None):
        """Initialize storage manager."""
        # Default library location
        if library_path is None:
            home = Path.home()
            library_path = home / "Papers" / "ArxivLibrary"
        
        self.library_path = Path(library_path)
        self.library = PaperLibrary(self.library_path)
        
        # Initialize clients
        self.arxiv_client = ArxivClient()
        self.extractor = SmartPDFExtractor()
        
        logger.info(f"Storage manager initialized with library at {self.library_path}")
    
    async def save_paper_to_library(
        self, 
        arxiv_id: str, 
        collection: Optional[str] = None,
        extraction_tier: str = "smart",
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Download paper, extract content, and save to library."""
        try:
            # Check if paper already exists
            existing_paper = self.library.get_paper(arxiv_id)
            if existing_paper:
                logger.info(f"Paper {arxiv_id} already exists in library")
                
                # Add to collection if specified
                if collection:
                    self.library.add_paper_to_collection(arxiv_id, collection)
                
                # Add tags if specified
                if tags:
                    self.library.add_tags(arxiv_id, tags)
                
                return {
                    "success": True,
                    "message": f"Paper {arxiv_id} already in library",
                    "paper": self._paper_to_dict(existing_paper),
                    "action": "updated"
                }
            
            # Get paper details from arXiv
            logger.info(f"Fetching paper details for {arxiv_id}")
            paper_details = await self.arxiv_client.get_paper_details(arxiv_id)
            
            if not paper_details:
                return {
                    "success": False,
                    "error": f"Could not fetch paper details for {arxiv_id}"
                }
            
            # Download PDF
            logger.info(f"Downloading PDF for {arxiv_id}")
            pdf_content = await self._download_pdf(arxiv_id)
            
            if not pdf_content:
                return {
                    "success": False,
                    "error": f"Could not download PDF for {arxiv_id}"
                }
            
            # Extract content
            logger.info(f"Extracting content from {arxiv_id}")
            extraction_result = await self._extract_paper_content(pdf_content, extraction_tier)
            
            # Create Paper object
            paper = Paper(
                arxiv_id=arxiv_id,
                title=paper_details.get('title', ''),
                authors=paper_details.get('authors', []),
                abstract=paper_details.get('summary', ''),
                published_date=paper_details.get('published', ''),
                tags=tags or [],
                notes=notes
            )
            
            # Save to library
            success = self.library.add_paper(
                paper=paper,
                pdf_content=pdf_content,
                extraction_result=extraction_result
            )
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to save paper {arxiv_id} to library"
                }
            
            # Add to collection if specified
            if collection:
                # Create collection if it doesn't exist
                await self._ensure_collection_exists(collection)
                self.library.add_paper_to_collection(arxiv_id, collection)
            
            logger.info(f"Successfully saved paper {arxiv_id} to library")
            
            return {
                "success": True,
                "message": f"Paper {arxiv_id} saved to library",
                "paper": self._paper_to_dict(paper),
                "extraction_info": {
                    "method": extraction_result.get("extraction_method", "unknown"),
                    "quality": extraction_result.get("quality_estimate", 0),
                    "content_length": len(extraction_result.get("content", ""))
                },
                "action": "saved"
            }
            
        except Exception as e:
            logger.error(f"Failed to save paper {arxiv_id}: {e}")
            return {
                "success": False,
                "error": f"Unexpected error saving paper {arxiv_id}: {str(e)}"
            }
    
    async def _download_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF content for paper."""
        try:
            # Construct arXiv PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        logger.error(f"Failed to download PDF: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {e}")
            return None
    
    async def _extract_paper_content(self, pdf_content: bytes, extraction_tier: str) -> Dict[str, Any]:
        """Extract content from PDF using the smart extractor."""
        try:
            # Save PDF to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_path = Path(tmp_file.name)
            
            try:
                # Extract using specified tier
                if extraction_tier.lower() == "premium":
                    from ..extraction.smart_extractor import ExtractionTier
                    result = await self.extractor.extract_paper(
                        pdf_path=tmp_path, 
                        user_preference=ExtractionTier.PREMIUM
                    )
                elif extraction_tier.lower() == "fast":
                    from ..extraction.smart_extractor import ExtractionTier
                    result = await self.extractor.extract_paper(
                        pdf_path=tmp_path, 
                        user_preference=ExtractionTier.FAST
                    )
                else:  # default to smart
                    from ..extraction.smart_extractor import ExtractionTier
                    result = await self.extractor.extract_paper(
                        pdf_path=tmp_path, 
                        user_preference=ExtractionTier.SMART
                    )
                
                # Return the extraction part of the result
                return result.get('extraction', result)
                
            finally:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Error extracting paper content: {e}")
            return {
                "content": "",
                "extraction_method": "failed",
                "quality_estimate": 0,
                "error": str(e)
            }
    
    async def _ensure_collection_exists(self, collection_name: str, description: str = "") -> bool:
        """Ensure collection exists, create if not."""
        collections = self.library.list_collections()
        existing_names = [c.name for c in collections]
        
        if collection_name not in existing_names:
            collection = Collection(
                name=collection_name,
                description=description or f"Collection for {collection_name}"
            )
            return self.library.create_collection(collection)
        
        return True
    
    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper object to dictionary."""
        return {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published_date": paper.published_date,
            "added_date": paper.added_date,
            "reading_status": paper.reading_status,
            "tags": paper.tags,
            "notes": paper.notes,
            "has_pdf": bool(paper.pdf_path and Path(paper.pdf_path).exists()),
            "has_extraction": bool(paper.extraction_path and Path(paper.extraction_path).exists())
        }
    
    def get_paper_from_library(self, arxiv_id: str, include_content: bool = False) -> Dict[str, Any]:
        """Get paper from library with optional content."""
        paper = self.library.get_paper(arxiv_id)
        
        if not paper:
            return {
                "success": False,
                "error": f"Paper {arxiv_id} not found in library"
            }
        
        result = {
            "success": True,
            "paper": self._paper_to_dict(paper)
        }
        
        # Include extracted content if requested
        if include_content and paper.extraction_path and Path(paper.extraction_path).exists():
            try:
                import json
                with open(paper.extraction_path, 'r') as f:
                    extraction_data = json.load(f)
                result["content"] = extraction_data
            except Exception as e:
                logger.error(f"Failed to load extraction data for {arxiv_id}: {e}")
                result["content_error"] = str(e)
        
        return result
    
    def list_library_papers(
        self, 
        collection: Optional[str] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List papers in library with filtering."""
        try:
            papers = self.library.list_papers(
                collection=collection,
                status=status,
                tags=tags,
                limit=limit,
                offset=offset
            )
            
            return {
                "success": True,
                "papers": [self._paper_to_dict(p) for p in papers],
                "count": len(papers),
                "filters": {
                    "collection": collection,
                    "status": status,
                    "tags": tags
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to list library papers: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_library(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search papers in library."""
        try:
            papers = self.library.search_papers(query, limit)
            
            return {
                "success": True,
                "papers": [self._paper_to_dict(p) for p in papers],
                "count": len(papers),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Failed to search library: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def manage_collections(self, action: str, name: str = "", description: str = "") -> Dict[str, Any]:
        """Manage collections (create, list)."""
        try:
            if action == "create":
                collection = Collection(name=name, description=description)
                success = self.library.create_collection(collection)
                
                return {
                    "success": success,
                    "message": f"Collection '{name}' {'created' if success else 'already exists'}"
                }
            
            elif action == "list":
                collections = self.library.list_collections()
                
                return {
                    "success": True,
                    "collections": [{
                        "name": c.name,
                        "description": c.description,
                        "created_date": c.created_date,
                        "paper_count": c.paper_count
                    } for c in collections]
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            logger.error(f"Failed to manage collections: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_paper_status(self, arxiv_id: str, status: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Update paper reading status and tags."""
        try:
            # Update status
            status_updated = self.library.update_reading_status(arxiv_id, status)
            
            # Add tags if provided
            tags_updated = True
            if tags:
                tags_updated = self.library.add_tags(arxiv_id, tags)
            
            if status_updated or tags_updated:
                return {
                    "success": True,
                    "message": f"Updated paper {arxiv_id}",
                    "status_updated": status_updated,
                    "tags_updated": tags_updated
                }
            else:
                return {
                    "success": False,
                    "error": f"Paper {arxiv_id} not found or update failed"
                }
                
        except Exception as e:
            logger.error(f"Failed to update paper {arxiv_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get library statistics and status."""
        try:
            stats = self.library.get_library_stats()
            
            return {
                "success": True,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get library stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }