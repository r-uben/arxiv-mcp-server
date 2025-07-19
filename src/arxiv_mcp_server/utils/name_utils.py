"""Comprehensive name normalization utilities for author searches."""

import re
import unicodedata
from typing import List, Set, Tuple
from itertools import combinations


class NameNormalizer:
    """Comprehensive author name normalization for robust ArXiv searches."""
    
    # Common academic name patterns
    TITLE_PREFIXES = {'dr', 'prof', 'professor', 'dr.', 'prof.'}
    SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv', 'jr.', 'sr.', 'ph.d', 'phd', 'md'}
    
    # Common accent mappings for academic names
    ACCENT_MAP = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ā': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e', 'ē': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i', 'ī': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'ō': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u', 'ū': 'u',
        'ñ': 'n', 'ç': 'c', 'ß': 'ss',
        'Á': 'A', 'À': 'A', 'Ä': 'A', 'Â': 'A', 'Ā': 'A', 'Ã': 'A',
        'É': 'E', 'È': 'E', 'Ë': 'E', 'Ê': 'E', 'Ē': 'E',
        'Í': 'I', 'Ì': 'I', 'Ï': 'I', 'Î': 'I', 'Ī': 'I',
        'Ó': 'O', 'Ò': 'O', 'Ö': 'O', 'Ô': 'O', 'Ō': 'O', 'Õ': 'O',
        'Ú': 'U', 'Ù': 'U', 'Ü': 'U', 'Û': 'U', 'Ū': 'U',
        'Ñ': 'N', 'Ç': 'C'
    }
    
    @classmethod
    def remove_accents(cls, text: str) -> str:
        """Remove accents using both manual mapping and Unicode normalization."""
        # First try manual mapping for common cases
        for accented, normal in cls.ACCENT_MAP.items():
            text = text.replace(accented, normal)
        
        # Then use Unicode normalization for anything missed
        normalized = unicodedata.normalize('NFD', text)
        ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return ascii_text
    
    @classmethod
    def clean_name_part(cls, name_part: str) -> str:
        """Clean a single name part (first/last name)."""
        name_part = name_part.strip()
        
        # Remove common punctuation but preserve dots for initials
        name_part = re.sub(r'[,;]', '', name_part)
        
        # Remove extra whitespace
        name_part = re.sub(r'\s+', ' ', name_part)
        
        return name_part
    
    @classmethod
    def parse_full_name(cls, full_name: str) -> Tuple[List[str], List[str]]:
        """Parse full name into first names and last names."""
        original_full_name = cls.clean_name_part(full_name)
        
        # Remove titles and suffixes while preserving original case
        words = original_full_name.split()
        cleaned_words = []
        for word in words:
            word_lower = word.lower().rstrip('.')
            if word_lower not in cls.TITLE_PREFIXES and word_lower not in cls.SUFFIXES:
                cleaned_words.append(word)
        
        if not cleaned_words:
            return [], []
        
        # Simple heuristic: last 1-2 words are surname, rest are given names
        if len(cleaned_words) == 1:
            return [], cleaned_words  # Single name treated as surname
        elif len(cleaned_words) == 2:
            return [cleaned_words[0]], [cleaned_words[1]]
        else:
            # For 3+ words, assume last word is surname, others are given names
            # But also consider compound surnames (last 2 words)
            given_names = cleaned_words[:-1]
            surnames = [cleaned_words[-1]]
            
            return given_names, surnames
    
    @classmethod
    def generate_name_variations(cls, full_name: str) -> List[str]:
        """Generate comprehensive variations of an author name."""
        variations = set()
        original = full_name.strip()
        variations.add(original)
        
        # Basic normalization
        cleaned = cls.clean_name_part(original)
        variations.add(cleaned)
        
        # Remove accents
        no_accents = cls.remove_accents(original)
        variations.add(no_accents)
        variations.add(cls.remove_accents(cleaned))
        
        # Parse name components
        given_names, surnames = cls.parse_full_name(original)
        
        if given_names or surnames:
            all_names = given_names + surnames
            
            # Generate various orderings and combinations
            cls._add_name_combinations(variations, given_names, surnames)
            cls._add_hyphenation_variations(variations, original, no_accents)
            cls._add_initial_variations(variations, given_names, surnames)
            cls._add_compound_name_variations(variations, all_names)
        
        # Remove empty strings and duplicates
        return list(filter(None, variations))
    
    @classmethod
    def _add_name_combinations(cls, variations: Set[str], given_names: List[str], surnames: List[str]):
        """Add different name order combinations."""
        if not given_names or not surnames:
            return
            
        # Standard: Given Surname
        standard = ' '.join(given_names + surnames)
        variations.add(standard)
        variations.add(cls.remove_accents(standard))
        
        # Reverse: Surname, Given (multiple formats)
        surname_str = ' '.join(surnames)
        given_str = ' '.join(given_names)
        
        reverse_comma = f"{surname_str}, {given_str}"
        variations.add(reverse_comma)
        variations.add(cls.remove_accents(reverse_comma))
        
        # Surname Given (no comma)
        surname_first = f"{surname_str} {given_str}"
        variations.add(surname_first)
        variations.add(cls.remove_accents(surname_first))
    
    @classmethod
    def _add_hyphenation_variations(cls, variations: Set[str], original: str, no_accents: str):
        """Add hyphenation and spacing variations."""
        for text in [original, no_accents]:
            # Space to hyphen
            if ' ' in text:
                hyphenated = text.replace(' ', '-')
                variations.add(hyphenated)
                
                # Also try selective hyphenation (last two words only)
                words = text.split()
                if len(words) >= 2:
                    selective_hyphen = ' '.join(words[:-2]) + ' ' + '-'.join(words[-2:])
                    variations.add(selective_hyphen)
            
            # Hyphen to space
            if '-' in text:
                spaced = text.replace('-', ' ')
                variations.add(spaced)
    
    @classmethod
    def _add_initial_variations(cls, variations: Set[str], given_names: List[str], surnames: List[str]):
        """Add variations with initials."""
        if not given_names or not surnames:
            return
            
        # First name initial + full surname
        if given_names:
            initial_first = given_names[0][0].upper() + '. ' + ' '.join(surnames)
            variations.add(initial_first)
            variations.add(cls.remove_accents(initial_first))
            
            # All initials + surname
            all_initials = '. '.join([name[0].upper() for name in given_names]) + '. ' + ' '.join(surnames)
            variations.add(all_initials)
            variations.add(cls.remove_accents(all_initials))
        
        # Surname + first initial
        if given_names:
            surname_initial = ' '.join(surnames) + ', ' + given_names[0][0].upper() + '.'
            variations.add(surname_initial)
            variations.add(cls.remove_accents(surname_initial))
    
    @classmethod
    def _add_compound_name_variations(cls, variations: Set[str], all_names: List[str]):
        """Add variations for compound surnames and given names."""
        if len(all_names) < 2:
            return
            
        # Try different ways of combining names
        for i in range(1, len(all_names)):
            # Split at different positions
            part1 = ' '.join(all_names[:i])
            part2 = ' '.join(all_names[i:])
            
            # Different separators
            variations.add(f"{part1} {part2}")
            variations.add(f"{part1}-{part2}")
            variations.add(f"{part2}, {part1}")
            variations.add(f"{part2} {part1}")
            
            # Remove accents from all
            variations.add(cls.remove_accents(f"{part1} {part2}"))
            variations.add(cls.remove_accents(f"{part1}-{part2}"))
            variations.add(cls.remove_accents(f"{part2}, {part1}"))
            variations.add(cls.remove_accents(f"{part2} {part1}"))


def detect_author_query(query: str) -> bool:
    """Detect if a query is likely searching for an author."""
    # Simple heuristics for author detection
    author_indicators = [
        'author:', 'by ', 'papers by', 'works by', 'publications by',
        'research by', 'articles by'
    ]
    
    query_lower = query.lower()
    
    # Check for explicit author indicators
    if any(indicator in query_lower for indicator in author_indicators):
        return True
    
    # Check if query looks like a name (2-4 words, mostly alphabetic)
    words = query.strip().split()
    if 2 <= len(words) <= 4:
        # Most words should be mostly alphabetic (allowing accents, hyphens, apostrophes)
        # Also check they're not common academic terms
        common_terms = {
            'machine', 'learning', 'deep', 'neural', 'network', 'networks', 
            'quantum', 'computing', 'artificial', 'intelligence', 'science',
            'algorithm', 'algorithms', 'model', 'models', 'analysis', 'theory',
            'method', 'methods', 'approach', 'system', 'systems', 'data'
        }
        
        name_like_words = 0
        non_academic_words = 0
        
        for word in words:
            word_clean = word.lower().strip('.,;!?')
            if re.match(r"^[a-zA-ZÀ-ÿ\-'\.]+$", word) and len(word) >= 2:
                name_like_words += 1
                if word_clean not in common_terms:
                    non_academic_words += 1
        
        # If most words look like name parts AND aren't common academic terms
        if (name_like_words >= len(words) * 0.7 and 
            non_academic_words >= len(words) * 0.5):
            return True
    
    return False


def generate_author_search_queries(query: str) -> List[str]:
    """Generate multiple search queries for author names."""
    queries = set()
    
    # Add original query
    queries.add(query)
    
    # If it looks like an author query, generate variations
    if detect_author_query(query):
        # Extract potential author name from query
        author_name = query
        
        # Remove common prefixes
        for prefix in ['author:', 'by ', 'papers by ', 'works by ', 'publications by ']:
            if author_name.lower().startswith(prefix):
                author_name = author_name[len(prefix):].strip()
                break
        
        # Generate name variations
        name_variations = NameNormalizer.generate_name_variations(author_name)
        
        # Create search queries for each variation
        for name_var in name_variations:
            # Direct name search
            queries.add(name_var)
            
            # Explicit author search
            queries.add(f"au:{name_var}")
            
            # Quoted name search (exact phrase)
            if ' ' in name_var:
                queries.add(f'"{name_var}"')
                queries.add(f'au:"{name_var}"')
    
    return list(queries)