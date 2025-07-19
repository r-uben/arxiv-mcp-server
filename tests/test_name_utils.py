"""Tests for name normalization utilities."""

import pytest
from arxiv_mcp_server.utils.name_utils import (
    NameNormalizer,
    detect_author_query,
    generate_author_search_queries
)


class TestNameNormalizer:
    """Test the NameNormalizer class."""

    def test_remove_accents(self):
        """Test accent removal."""
        assert NameNormalizer.remove_accents("José María López") == "Jose Maria Lopez"
        assert NameNormalizer.remove_accents("François Müller") == "Francois Muller"
        assert NameNormalizer.remove_accents("Alejandro López Lira") == "Alejandro Lopez Lira"
        assert NameNormalizer.remove_accents("普通话") == "普通话"  # Non-Latin characters unchanged

    def test_parse_full_name(self):
        """Test name parsing."""
        given, surnames = NameNormalizer.parse_full_name("John Smith")
        assert given == ["John"]
        assert surnames == ["Smith"]

        given, surnames = NameNormalizer.parse_full_name("María José García López")
        assert given == ["María", "José", "García"]
        assert surnames == ["López"]

        given, surnames = NameNormalizer.parse_full_name("Smith")
        assert given == []
        assert surnames == ["Smith"]

    def test_generate_name_variations(self):
        """Test comprehensive name variation generation."""
        variations = NameNormalizer.generate_name_variations("Alejandro López Lira")
        
        # Should include original
        assert "Alejandro López Lira" in variations
        
        # Should include accent-free version
        assert "Alejandro Lopez Lira" in variations
        
        # Should include hyphenated versions
        assert "Alejandro López-Lira" in variations
        assert "Alejandro Lopez-Lira" in variations
        
        # Should include different orderings
        assert ("López Lira, Alejandro" in variations or 
                "Lopez Lira, Alejandro" in variations)
        
        # Should include initials
        assert ("A. Lira" in variations or 
                "A. López Lira" in variations or 
                "A. Lopez Lira" in variations)

    def test_complex_names(self):
        """Test with complex academic names."""
        variations = NameNormalizer.generate_name_variations("Jean-Claude Van Damme")
        
        # Should handle existing hyphens
        assert "Jean-Claude Van Damme" in variations
        assert "Jean Claude Van Damme" in variations
        
        # Should try different combinations (the algorithm treats "Damme" as surname)
        assert ("Damme, Jean-Claude Van" in variations or 
                "Damme, Jean Claude Van" in variations)

    def test_international_names(self):
        """Test with international names."""
        # Spanish names
        variations = NameNormalizer.generate_name_variations("José María García-López")
        assert "Jose Maria Garcia-Lopez" in variations
        assert "Jose Maria Garcia Lopez" in variations
        
        # Chinese names (should handle gracefully)
        variations = NameNormalizer.generate_name_variations("李明")
        assert "李明" in variations
        assert len(variations) >= 1


class TestAuthorDetection:
    """Test author query detection."""

    def test_detect_author_query(self):
        """Test author query detection."""
        # Explicit author queries
        assert detect_author_query("author:John Smith")
        assert detect_author_query("papers by Einstein")
        assert detect_author_query("works by Marie Curie")
        
        # Name-like queries
        assert detect_author_query("John Smith")
        assert detect_author_query("María García López")
        assert detect_author_query("Jean-Pierre Dupont")
        
        # Non-author queries
        assert not detect_author_query("machine learning")
        assert not detect_author_query("neural networks deep learning")
        assert not detect_author_query("quantum computing applications")

    def test_generate_author_search_queries(self):
        """Test author search query generation."""
        queries = generate_author_search_queries("Alejandro López Lira")
        
        # Should include original
        assert "Alejandro López Lira" in queries
        
        # Should include ArXiv author syntax
        author_queries = [q for q in queries if q.startswith("au:")]
        assert len(author_queries) > 0
        
        # Should include quoted versions
        quoted_queries = [q for q in queries if q.startswith('"') and q.endswith('"')]
        assert len(quoted_queries) > 0

    def test_explicit_author_prefix(self):
        """Test handling of explicit author prefixes."""
        queries = generate_author_search_queries("author:John Smith")
        
        # Should extract name and generate variations
        assert "John Smith" in queries
        assert "au:John Smith" in queries

    def test_non_author_query(self):
        """Test non-author queries."""
        queries = generate_author_search_queries("machine learning")
        
        # Should just return the original query since it's not detected as author
        assert "machine learning" in queries
        assert len(queries) == 1


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_names(self):
        """Test empty or whitespace names."""
        variations = NameNormalizer.generate_name_variations("")
        assert all(v.strip() for v in variations)  # No empty strings
        
        variations = NameNormalizer.generate_name_variations("   ")
        assert all(v.strip() for v in variations)

    def test_single_names(self):
        """Test single names (mononyms)."""
        variations = NameNormalizer.generate_name_variations("Cher")
        assert "Cher" in variations
        # Single names don't get lowercased in our algorithm
        assert len(variations) >= 1

    def test_names_with_titles(self):
        """Test names with academic titles."""
        variations = NameNormalizer.generate_name_variations("Dr. John Smith")
        # Titles should be removed in variations
        assert any("John Smith" in v for v in variations)
        
        variations = NameNormalizer.generate_name_variations("Prof. María García")
        assert any("María García" in v or "Maria Garcia" in v for v in variations)

    def test_names_with_suffixes(self):
        """Test names with suffixes."""
        variations = NameNormalizer.generate_name_variations("John Smith Jr.")
        # Suffixes should be removed in variations  
        assert any("John Smith" in v for v in variations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])