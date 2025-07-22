"""Paper library management with SQLite backend."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Paper metadata model."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: Optional[str] = None
    pdf_path: Optional[str] = None
    extraction_path: Optional[str] = None
    added_date: Optional[str] = None
    reading_status: str = "unread"  # unread, reading, completed
    tags: List[str] = None
    notes: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.added_date is None:
            self.added_date = datetime.now().isoformat()


@dataclass
class Collection:
    """Collection model for organizing papers."""
    name: str
    description: str = ""
    created_date: Optional[str] = None
    paper_count: int = 0
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()


class PaperLibrary:
    """SQLite-based paper library management."""
    
    def __init__(self, library_path: Path):
        """Initialize paper library with database path."""
        self.library_path = library_path
        self.library_path.mkdir(parents=True, exist_ok=True)
        
        # Database and storage paths
        self.db_path = library_path / "library.db"
        self.pdfs_path = library_path / "pdfs"
        self.extractions_path = library_path / "extractions"
        
        # Create directories
        self.pdfs_path.mkdir(exist_ok=True)
        self.extractions_path.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Papers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,  -- JSON array
                abstract TEXT,
                published_date TEXT,
                pdf_path TEXT,
                extraction_path TEXT,
                added_date TEXT NOT NULL,
                reading_status TEXT DEFAULT 'unread',
                tags TEXT,  -- JSON array
                notes TEXT DEFAULT ''
            )
        ''')
        
        # Collections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                created_date TEXT NOT NULL
            )
        ''')
        
        # Paper-Collection mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_collections (
                arxiv_id TEXT,
                collection_name TEXT,
                added_date TEXT NOT NULL,
                PRIMARY KEY (arxiv_id, collection_name),
                FOREIGN KEY (arxiv_id) REFERENCES papers (arxiv_id),
                FOREIGN KEY (collection_name) REFERENCES collections (name)
            )
        ''')
        
        # Citations table (for citation following)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citations (
                citing_paper TEXT,
                cited_paper TEXT,
                citation_text TEXT,
                resolved_arxiv_id TEXT,
                added_date TEXT NOT NULL,
                PRIMARY KEY (citing_paper, cited_paper),
                FOREIGN KEY (citing_paper) REFERENCES papers (arxiv_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_status ON papers (reading_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_added_date ON papers (added_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations (citing_paper)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations (cited_paper)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Paper library initialized at {self.library_path}")
    
    def add_paper(self, paper: Paper, pdf_content: Optional[bytes] = None, 
                  extraction_result: Optional[Dict[str, Any]] = None) -> bool:
        """Add paper to library with optional PDF and extraction data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save PDF if provided
            pdf_path = None
            if pdf_content:
                pdf_filename = f"{paper.arxiv_id}.pdf"
                pdf_path = self.pdfs_path / pdf_filename
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_content)
                paper.pdf_path = str(pdf_path)
            
            # Save extraction result if provided
            extraction_path = None
            if extraction_result:
                extraction_filename = f"{paper.arxiv_id}.json"
                extraction_path = self.extractions_path / extraction_filename
                with open(extraction_path, 'w') as f:
                    json.dump(extraction_result, f, indent=2)
                paper.extraction_path = str(extraction_path)
            
            # Insert paper into database
            cursor.execute('''
                INSERT OR REPLACE INTO papers 
                (arxiv_id, title, authors, abstract, published_date, pdf_path, 
                 extraction_path, added_date, reading_status, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper.arxiv_id,
                paper.title,
                json.dumps(paper.authors),
                paper.abstract,
                paper.published_date,
                paper.pdf_path,
                paper.extraction_path,
                paper.added_date,
                paper.reading_status,
                json.dumps(paper.tags),
                paper.notes
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added paper {paper.arxiv_id} to library")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper {paper.arxiv_id}: {e}")
            return False
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Get paper by arXiv ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM papers WHERE arxiv_id = ?', (arxiv_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return Paper(
                    arxiv_id=row[0],
                    title=row[1],
                    authors=json.loads(row[2]),
                    abstract=row[3],
                    published_date=row[4],
                    pdf_path=row[5],
                    extraction_path=row[6],
                    added_date=row[7],
                    reading_status=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    notes=row[10]
                )
            return None
            
        except Exception as e:
            logger.error(f"Failed to get paper {arxiv_id}: {e}")
            return None
    
    def list_papers(self, collection: Optional[str] = None, 
                   status: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   limit: int = 100, offset: int = 0) -> List[Paper]:
        """List papers with optional filtering."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build query with filters
            query = "SELECT DISTINCT p.* FROM papers p"
            conditions = []
            params = []
            
            # Filter by collection
            if collection:
                query += " JOIN paper_collections pc ON p.arxiv_id = pc.arxiv_id"
                conditions.append("pc.collection_name = ?")
                params.append(collection)
            
            # Filter by status
            if status:
                conditions.append("p.reading_status = ?")
                params.append(status)
            
            # Filter by tags (papers that have ALL specified tags)
            if tags:
                for tag in tags:
                    conditions.append("JSON_EXTRACT(p.tags, '$') LIKE ?")
                    params.append(f'%"{tag}"%')
            
            # Add WHERE clause if needed
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            # Order and limit
            query += " ORDER BY p.added_date DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            papers = []
            for row in rows:
                papers.append(Paper(
                    arxiv_id=row[0],
                    title=row[1],
                    authors=json.loads(row[2]),
                    abstract=row[3],
                    published_date=row[4],
                    pdf_path=row[5],
                    extraction_path=row[6],
                    added_date=row[7],
                    reading_status=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    notes=row[10]
                ))
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            return []
    
    def search_papers(self, query: str, limit: int = 50) -> List[Paper]:
        """Full-text search across papers."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Search in title, abstract, authors, and notes
            search_query = '''
                SELECT * FROM papers 
                WHERE title LIKE ? OR abstract LIKE ? OR authors LIKE ? OR notes LIKE ?
                ORDER BY added_date DESC 
                LIMIT ?
            '''
            
            search_term = f"%{query}%"
            cursor.execute(search_query, (search_term, search_term, search_term, search_term, limit))
            rows = cursor.fetchall()
            
            conn.close()
            
            papers = []
            for row in rows:
                papers.append(Paper(
                    arxiv_id=row[0],
                    title=row[1],
                    authors=json.loads(row[2]),
                    abstract=row[3],
                    published_date=row[4],
                    pdf_path=row[5],
                    extraction_path=row[6],
                    added_date=row[7],
                    reading_status=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    notes=row[10]
                ))
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to search papers: {e}")
            return []
    
    def create_collection(self, collection: Collection) -> bool:
        """Create a new collection."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO collections (name, description, created_date)
                VALUES (?, ?, ?)
            ''', (collection.name, collection.description, collection.created_date))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created collection: {collection.name}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Collection {collection.name} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to create collection {collection.name}: {e}")
            return False
    
    def add_paper_to_collection(self, arxiv_id: str, collection_name: str) -> bool:
        """Add paper to collection."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO paper_collections (arxiv_id, collection_name, added_date)
                VALUES (?, ?, ?)
            ''', (arxiv_id, collection_name, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added paper {arxiv_id} to collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper {arxiv_id} to collection {collection_name}: {e}")
            return False
    
    def list_collections(self) -> List[Collection]:
        """List all collections with paper counts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.name, c.description, c.created_date, COUNT(pc.arxiv_id) as paper_count
                FROM collections c
                LEFT JOIN paper_collections pc ON c.name = pc.collection_name
                GROUP BY c.name, c.description, c.created_date
                ORDER BY c.created_date DESC
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            collections = []
            for row in rows:
                collections.append(Collection(
                    name=row[0],
                    description=row[1],
                    created_date=row[2],
                    paper_count=row[3]
                ))
            
            return collections
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def update_reading_status(self, arxiv_id: str, status: str) -> bool:
        """Update paper reading status."""
        if status not in ['unread', 'reading', 'completed']:
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE papers SET reading_status = ? WHERE arxiv_id = ?
            ''', (status, arxiv_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated reading status for {arxiv_id} to {status}")
            return cursor.rowcount > 0
            
        except Exception as e:
            logger.error(f"Failed to update reading status for {arxiv_id}: {e}")
            return False
    
    def add_tags(self, arxiv_id: str, new_tags: List[str]) -> bool:
        """Add tags to paper."""
        try:
            paper = self.get_paper(arxiv_id)
            if not paper:
                return False
            
            # Merge tags
            existing_tags = set(paper.tags)
            existing_tags.update(new_tags)
            updated_tags = list(existing_tags)
            
            # Update in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE papers SET tags = ? WHERE arxiv_id = ?
            ''', (json.dumps(updated_tags), arxiv_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added tags {new_tags} to paper {arxiv_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add tags to {arxiv_id}: {e}")
            return False
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total papers
            cursor.execute('SELECT COUNT(*) FROM papers')
            total_papers = cursor.fetchone()[0]
            
            # Papers by status
            cursor.execute('''
                SELECT reading_status, COUNT(*) 
                FROM papers 
                GROUP BY reading_status
            ''')
            status_counts = dict(cursor.fetchall())
            
            # Total collections
            cursor.execute('SELECT COUNT(*) FROM collections')
            total_collections = cursor.fetchone()[0]
            
            # Storage usage
            pdfs_size = sum(f.stat().st_size for f in self.pdfs_path.glob('*.pdf') if f.is_file())
            extractions_size = sum(f.stat().st_size for f in self.extractions_path.glob('*.json') if f.is_file())
            
            conn.close()
            
            return {
                'total_papers': total_papers,
                'status_counts': status_counts,
                'total_collections': total_collections,
                'storage_mb': {
                    'pdfs': round(pdfs_size / (1024 * 1024), 2),
                    'extractions': round(extractions_size / (1024 * 1024), 2)
                },
                'library_path': str(self.library_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get library stats: {e}")
            return {}