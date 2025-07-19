"""Advanced paper analysis and comparison tools."""

import re
import logging
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Set
from urllib.parse import quote

from ..extraction.paper_reader import PaperReader, PaperAnalyzer

logger = logging.getLogger(__name__)


class PaperComparator:
    """Compare multiple papers and find relationships."""
    
    def __init__(self):
        self.analyzer = PaperAnalyzer()
        self.reader = PaperReader()
    
    async def compare_papers(
        self, 
        paper_ids: List[str], 
        comparison_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """Compare multiple papers across different aspects."""
        
        if comparison_aspects is None:
            comparison_aspects = ["methodology", "results", "contributions", "scope"]
        
        # Get content for all papers
        papers_content = {}
        papers_summaries = {}
        
        for paper_id in paper_ids:
            try:
                summary = await self.analyzer.summarize_paper(paper_id)
                if "error" not in summary:
                    papers_summaries[paper_id] = summary
                    
                findings = await self.analyzer.extract_key_findings(paper_id)
                if "error" not in findings:
                    papers_content[paper_id] = findings
                    
            except Exception as e:
                logger.warning(f"Failed to analyze paper {paper_id}: {e}")
                continue
        
        if not papers_content:
            return {"error": "Could not analyze any of the provided papers"}
        
        # Perform comparison
        comparison = {
            "papers_analyzed": list(papers_content.keys()),
            "comparison_aspects": comparison_aspects,
            "overview": self._generate_comparison_overview(papers_summaries),
            "detailed_comparison": {}
        }
        
        # Compare each aspect
        for aspect in comparison_aspects:
            if aspect == "methodology":
                comparison["detailed_comparison"]["methodology"] = self._compare_methodologies(papers_content)
            elif aspect == "results":
                comparison["detailed_comparison"]["results"] = self._compare_results(papers_content)
            elif aspect == "contributions":
                comparison["detailed_comparison"]["contributions"] = self._compare_contributions(papers_content)
            elif aspect == "scope":
                comparison["detailed_comparison"]["scope"] = self._compare_scope(papers_summaries)
        
        # Find similarities and differences
        comparison["similarities"] = self._find_similarities(papers_content)
        comparison["differences"] = self._find_differences(papers_content)
        comparison["complementary_aspects"] = self._find_complementary_aspects(papers_content)
        
        return comparison
    
    def _generate_comparison_overview(self, papers_summaries: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate overview comparison statistics."""
        overview = {
            "paper_count": len(papers_summaries),
            "avg_page_count": 0,
            "avg_word_count": 0,
            "common_sections": set(),
            "unique_sections": {}
        }
        
        if not papers_summaries:
            return overview
        
        total_pages = sum(p.get("metadata", {}).get("page_count", 0) for p in papers_summaries.values())
        total_words = sum(p.get("metadata", {}).get("word_count", 0) for p in papers_summaries.values())
        
        overview["avg_page_count"] = total_pages / len(papers_summaries)
        overview["avg_word_count"] = total_words / len(papers_summaries)
        
        # Find common sections
        all_sections = [set(p.get("sections_found", [])) for p in papers_summaries.values()]
        if all_sections:
            overview["common_sections"] = set.intersection(*all_sections)
            
            # Find unique sections per paper
            for paper_id, summary in papers_summaries.items():
                sections = set(summary.get("sections_found", []))
                unique = sections - overview["common_sections"]
                if unique:
                    overview["unique_sections"][paper_id] = list(unique)
        
        return overview
    
    def _compare_methodologies(self, papers_content: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare methodologies across papers."""
        methodologies = {}
        
        for paper_id, content in papers_content.items():
            methodology = content.get("methodology", {})
            if methodology:
                methodologies[paper_id] = {
                    "approach": methodology.get("approach", ""),
                    "length": methodology.get("length", 0),
                    "algorithms_mentioned": methodology.get("algorithms_mentioned", 0)
                }
        
        if not methodologies:
            return {"note": "No methodology sections found"}
        
        # Find common keywords in methodologies
        all_texts = [m["approach"].lower() for m in methodologies.values()]
        common_keywords = self._find_common_keywords(all_texts)
        
        return {
            "methodologies_by_paper": methodologies,
            "common_approaches": common_keywords,
            "methodology_complexity": {
                paper_id: {
                    "length_category": "short" if m["length"] < 200 else "medium" if m["length"] < 500 else "long",
                    "algorithm_focus": m["algorithms_mentioned"] > 2
                }
                for paper_id, m in methodologies.items()
            }
        }
    
    def _compare_results(self, papers_content: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare results across papers."""
        results_comparison = {
            "results_by_paper": {},
            "performance_metrics": {},
            "common_evaluation_approaches": []
        }
        
        for paper_id, content in papers_content.items():
            results = content.get("results", [])
            if results:
                results_comparison["results_by_paper"][paper_id] = results
                
                # Extract performance numbers
                metrics = self._extract_performance_metrics(results)
                if metrics:
                    results_comparison["performance_metrics"][paper_id] = metrics
        
        return results_comparison
    
    def _compare_contributions(self, papers_content: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare contributions across papers."""
        contributions_analysis = {
            "contributions_by_paper": {},
            "contribution_types": {},
            "novel_aspects": {}
        }
        
        for paper_id, content in papers_content.items():
            contributions = content.get("contributions", [])
            if contributions:
                contributions_analysis["contributions_by_paper"][paper_id] = contributions
                
                # Categorize contribution types
                contrib_types = self._categorize_contributions(contributions)
                contributions_analysis["contribution_types"][paper_id] = contrib_types
                
                # Find novel aspects
                novel = self._extract_novel_aspects(contributions)
                if novel:
                    contributions_analysis["novel_aspects"][paper_id] = novel
        
        return contributions_analysis
    
    def _compare_scope(self, papers_summaries: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare scope and coverage of papers."""
        scope_comparison = {
            "scope_by_paper": {},
            "coverage_overlap": {},
            "complementary_coverage": {}
        }
        
        for paper_id, summary in papers_summaries.items():
            abstract = summary.get("abstract", "")
            if abstract and abstract != "Not found":
                scope_comparison["scope_by_paper"][paper_id] = {
                    "abstract_length": len(abstract.split()),
                    "key_topics": self._extract_key_topics(abstract),
                    "research_domain": self._infer_research_domain(abstract)
                }
        
        return scope_comparison
    
    def _find_similarities(self, papers_content: Dict[str, Dict]) -> List[str]:
        """Find similarities between papers."""
        similarities = []
        
        # Check for similar methodologies
        methodologies = [content.get("methodology", {}).get("approach", "") for content in papers_content.values()]
        if len(methodologies) > 1:
            common_method_keywords = self._find_common_keywords(methodologies, min_frequency=len(methodologies))
            if common_method_keywords:
                similarities.append(f"Common methodological approaches: {', '.join(common_method_keywords[:3])}")
        
        # Check for similar equation types
        all_equations = []
        for content in papers_content.values():
            all_equations.extend(content.get("equations", []))
        
        if len(all_equations) > 1:
            similarities.append(f"Mathematical formulations present in multiple papers")
        
        return similarities
    
    def _find_differences(self, papers_content: Dict[str, Dict]) -> List[str]:
        """Find key differences between papers."""
        differences = []
        
        # Check methodology differences
        methodologies = {pid: content.get("methodology", {}) for pid, content in papers_content.items()}
        method_lengths = [m.get("length", 0) for m in methodologies.values() if m]
        
        if method_lengths and max(method_lengths) > 2 * min(method_lengths):
            differences.append("Significant variation in methodology complexity")
        
        # Check result presentation differences
        results = {pid: len(content.get("results", [])) for pid, content in papers_content.items()}
        if results and max(results.values()) > 2 * min(results.values()):
            differences.append("Different levels of experimental validation")
        
        return differences
    
    def _find_complementary_aspects(self, papers_content: Dict[str, Dict]) -> List[str]:
        """Find how papers complement each other."""
        complementary = []
        
        # Find papers with different strengths
        theoretical_papers = []
        experimental_papers = []
        
        for paper_id, content in papers_content.items():
            equations = len(content.get("equations", []))
            results = len(content.get("results", []))
            
            if equations > results:
                theoretical_papers.append(paper_id)
            elif results > equations:
                experimental_papers.append(paper_id)
        
        if theoretical_papers and experimental_papers:
            complementary.append(f"Theoretical ({len(theoretical_papers)}) and experimental ({len(experimental_papers)}) approaches complement each other")
        
        return complementary
    
    def _find_common_keywords(self, texts: List[str], min_frequency: int = 2) -> List[str]:
        """Find common keywords across texts."""
        if not texts:
            return []
        
        # Simple keyword extraction
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            all_words.extend(words)
        
        # Count frequency
        word_count = {}
        for word in all_words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Filter common technical words and return frequent ones
        common_keywords = [
            word for word, count in word_count.items() 
            if count >= min_frequency and word not in {
                'paper', 'method', 'approach', 'result', 'show', 'present', 
                'propose', 'study', 'research', 'analysis', 'data', 'model'
            }
        ]
        
        return sorted(common_keywords, key=lambda w: word_count[w], reverse=True)
    
    def _extract_performance_metrics(self, results: List[str]) -> List[Dict[str, Any]]:
        """Extract performance metrics from results."""
        metrics = []
        
        for result in results:
            # Look for percentage improvements
            percentage_matches = re.findall(r'(\d+\.?\d*)%', result)
            if percentage_matches:
                metrics.append({
                    "type": "percentage",
                    "values": [float(m) for m in percentage_matches],
                    "context": result[:100]
                })
            
            # Look for numerical comparisons
            number_matches = re.findall(r'(\d+\.?\d+)', result)
            if number_matches:
                metrics.append({
                    "type": "numerical",
                    "values": [float(m) for m in number_matches],
                    "context": result[:100]
                })
        
        return metrics
    
    def _categorize_contributions(self, contributions: List[str]) -> List[str]:
        """Categorize types of contributions."""
        categories = []
        
        for contrib in contributions:
            contrib_lower = contrib.lower()
            if any(word in contrib_lower for word in ['algorithm', 'method', 'approach']):
                categories.append("methodological")
            elif any(word in contrib_lower for word in ['theory', 'theoretical', 'proof']):
                categories.append("theoretical")
            elif any(word in contrib_lower for word in ['empirical', 'experimental', 'evaluation']):
                categories.append("empirical")
            elif any(word in contrib_lower for word in ['dataset', 'benchmark', 'corpus']):
                categories.append("data")
            else:
                categories.append("other")
        
        return list(set(categories))
    
    def _extract_novel_aspects(self, contributions: List[str]) -> List[str]:
        """Extract novel aspects from contributions."""
        novel_aspects = []
        
        for contrib in contributions:
            if any(word in contrib.lower() for word in ['novel', 'new', 'first', 'innovative']):
                novel_aspects.append(contrib[:100] + "..." if len(contrib) > 100 else contrib)
        
        return novel_aspects
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text."""
        # Simple topic extraction based on common academic keywords
        topic_patterns = [
            r'\b(machine learning|deep learning|neural networks?|artificial intelligence|AI)\b',
            r'\b(computer vision|image processing|computer graphics)\b',
            r'\b(natural language processing|NLP|language models?)\b',
            r'\b(optimization|algorithm|computational)\b',
            r'\b(statistics|statistical|probability|probabilistic)\b',
            r'\b(robotics|autonomous|control)\b',
        ]
        
        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            topics.extend(matches)
        
        return list(set(topics))
    
    def _infer_research_domain(self, text: str) -> str:
        """Infer research domain from abstract."""
        domains = {
            "computer_science": ['algorithm', 'computation', 'software', 'programming'],
            "machine_learning": ['learning', 'training', 'model', 'neural', 'classification'],
            "mathematics": ['theorem', 'proof', 'mathematical', 'equation', 'formula'],
            "physics": ['physical', 'quantum', 'particle', 'energy', 'field'],
            "biology": ['biological', 'genetic', 'protein', 'cell', 'organism']
        }
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"


class CitationTracker:
    """Track citations and paper relationships."""
    
    def __init__(self):
        self.base_url = "https://api.semanticscholar.org/graph/v1"
    
    async def find_citing_papers(
        self, 
        arxiv_id: str, 
        max_results: int = 20
    ) -> Dict[str, Any]:
        """Find papers that cite the given ArXiv paper."""
        
        try:
            # First, get the Semantic Scholar paper ID
            s2_paper_id = await self._get_s2_paper_id(arxiv_id)
            if not s2_paper_id:
                return {"error": f"Paper {arxiv_id} not found in Semantic Scholar"}
            
            # Get citing papers
            citing_papers = await self._get_citing_papers(s2_paper_id, max_results)
            
            return {
                "arxiv_id": arxiv_id,
                "s2_paper_id": s2_paper_id,
                "citing_papers": citing_papers,
                "citation_count": len(citing_papers),
                "most_recent_citations": sorted(
                    citing_papers, 
                    key=lambda p: p.get("year", 0), 
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            logger.error(f"Error finding citing papers: {e}")
            return {"error": str(e)}
    
    async def get_citation_network(
        self, 
        arxiv_id: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """Build a citation network around a paper."""
        
        network = {
            "center_paper": arxiv_id,
            "depth": depth,
            "nodes": {},
            "edges": [],
            "statistics": {}
        }
        
        try:
            # Start with the center paper
            s2_paper_id = await self._get_s2_paper_id(arxiv_id)
            if not s2_paper_id:
                return {"error": f"Paper {arxiv_id} not found"}
            
            # Get paper details
            paper_details = await self._get_paper_details(s2_paper_id)
            network["nodes"][arxiv_id] = paper_details
            
            # Get papers this paper cites (references)
            references = await self._get_references(s2_paper_id, limit=10)
            for ref in references:
                if ref.get("externalIds", {}).get("ArXiv"):
                    ref_arxiv_id = ref["externalIds"]["ArXiv"]
                    network["nodes"][ref_arxiv_id] = ref
                    network["edges"].append({
                        "from": arxiv_id,
                        "to": ref_arxiv_id,
                        "type": "cites"
                    })
            
            # Get papers that cite this paper
            citing = await self._get_citing_papers(s2_paper_id, limit=10)
            for cite in citing:
                if cite.get("externalIds", {}).get("ArXiv"):
                    cite_arxiv_id = cite["externalIds"]["ArXiv"]
                    network["nodes"][cite_arxiv_id] = cite
                    network["edges"].append({
                        "from": cite_arxiv_id,
                        "to": arxiv_id,
                        "type": "cites"
                    })
            
            # Calculate statistics
            network["statistics"] = {
                "total_nodes": len(network["nodes"]),
                "total_edges": len(network["edges"]),
                "references_count": len(references),
                "citations_count": len(citing),
                "arxiv_papers_in_network": len([n for n in network["nodes"] if n != arxiv_id])
            }
            
        except Exception as e:
            logger.error(f"Error building citation network: {e}")
            network["error"] = str(e)
        
        return network
    
    async def _get_s2_paper_id(self, arxiv_id: str) -> Optional[str]:
        """Get Semantic Scholar paper ID from ArXiv ID."""
        url = f"{self.base_url}/paper/arXiv:{arxiv_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("paperId")
                else:
                    logger.warning(f"Paper not found in Semantic Scholar: {arxiv_id}")
                    return None
    
    async def _get_paper_details(self, s2_paper_id: str) -> Dict[str, Any]:
        """Get paper details from Semantic Scholar."""
        url = f"{self.base_url}/paper/{s2_paper_id}"
        params = {"fields": "title,authors,year,abstract,citationCount,referenceCount,externalIds"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
    
    async def _get_citing_papers(self, s2_paper_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that cite this paper."""
        url = f"{self.base_url}/paper/{s2_paper_id}/citations"
        params = {
            "fields": "title,authors,year,abstract,externalIds,citationCount",
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [citation["citingPaper"] for citation in data.get("data", [])]
                else:
                    return []
    
    async def _get_references(self, s2_paper_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get papers that this paper references."""
        url = f"{self.base_url}/paper/{s2_paper_id}/references"
        params = {
            "fields": "title,authors,year,abstract,externalIds,citationCount",
            "limit": limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [ref["citedPaper"] for ref in data.get("data", [])]
                else:
                    return []