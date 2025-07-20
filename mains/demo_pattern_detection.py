#!/usr/bin/env python3
"""Demonstrate how pattern detection works in the ArXiv MCP server."""

import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def demo_pattern_detection():
    """Show exactly how patterns are detected in PDF text."""
    
    print("🔍 Pattern Detection Demo for PDF Difficulty Assessment\n")
    
    # Patterns from the actual system
    math_patterns = [
        r'\$.*?\$',           # Inline math
        r'\$\$.*?\$\$',       # Display math
        r'\\begin\{equation\}', r'\\begin\{align\}',
        r'∫', r'∑', r'∏', r'√', r'∂', r'∇',  # Math symbols
        r'α', r'β', r'γ', r'δ', r'θ', r'λ', r'μ', r'π', r'σ', r'φ'  # Greek letters
    ]
    
    complex_layout_patterns = [
        r'Figure\s+\d+', r'Table\s+\d+',  # Figures/tables
        r'\btwo.?column\b', r'\bmulti.?column\b',  # Multi-column
        r'\\includegraphics', r'\\begin\{figure\}', r'\\begin\{table\}'  # LaTeX elements
    ]
    
    # Sample text from different types of documents
    samples = {
        "Simple Text": """
        This is a simple document with basic text. 
        It contains regular sentences and paragraphs.
        No mathematical content or complex layouts here.
        """,
        
        "Math Heavy": """
        The equation $E = mc^2$ shows energy-mass equivalence.
        For integration: ∫ f(x)dx and summation: ∑ᵢ xᵢ
        Greek letters are common: α, β, θ, λ, π, σ
        Display math: $$\\int_{0}^{\\infty} e^{-x} dx = 1$$
        """,
        
        "Academic Paper": """
        Figure 1 shows the experimental setup used in this study.
        Table 2 presents the results of our analysis.
        The two-column layout improves readability.
        \\begin{equation} x = \\frac{-b ± √(b² - 4ac)}{2a} \\end{equation}
        See \\includegraphics{diagram.png} for details.
        """,
        
        "Corrupted OCR": """
        Th�s t�xt h�s m�ny c�rrupt�d ch�r�ct�rs.
        �������������������������������������
        Some weird chars: ┌┐└┘│─╔╗╚╝║═⌐¬
        Normal text mixed with garbage: ♠♣♥♦◊○●
        """
    }
    
    print("="*80)
    print("PATTERN DETECTION ANALYSIS")
    print("="*80)
    
    for doc_type, text in samples.items():
        print(f"\n📄 **{doc_type}**")
        print("-" * 40)
        
        # 1. Math Density Calculation
        math_matches = 0
        math_found = []
        
        for pattern in math_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                math_matches += len(matches)
                math_found.extend([f"{pattern}: {matches}"])
        
        text_length = len(text.split())
        math_density = min(math_matches / max(text_length, 1) * 100, 1.0)
        
        print(f"🧮 **Math Detection:**")
        print(f"   Math patterns found: {math_matches}")
        print(f"   Text length: {text_length} words")
        print(f"   Math density: {math_density:.1%}")
        if math_found:
            for finding in math_found[:3]:  # Show first 3
                print(f"   Found: {finding}")
        
        # 2. Layout Complexity
        complexity_indicators = 0
        layout_found = []
        
        for pattern in complex_layout_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                complexity_indicators += len(matches)
                layout_found.append(f"{pattern}: {matches}")
        
        # Multi-column detection (short lines)
        lines = text.split('\n')
        short_lines = sum(1 for line in lines if len(line.strip()) < 50)
        short_line_ratio = short_lines / max(len(lines), 1)
        
        if short_line_ratio > 0.3:
            complexity_indicators += 5
            layout_found.append(f"Multi-column detected: {short_line_ratio:.1%} short lines")
        
        layout_complexity = min(complexity_indicators / 10.0, 1.0)
        
        print(f"📐 **Layout Detection:**")
        print(f"   Complexity indicators: {complexity_indicators}")
        print(f"   Layout complexity: {layout_complexity:.1%}")
        print(f"   Short line ratio: {short_line_ratio:.1%}")
        if layout_found:
            for finding in layout_found:
                print(f"   Found: {finding}")
        
        # 3. Text Extractability
        total_chars = len(text)
        garbled_chars = text.count('�')  # Replacement character
        weird_chars = len(re.findall(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', text))
        
        problem_ratio = (garbled_chars + weird_chars) / max(total_chars, 1)
        extractability = max(0.0, 1.0 - problem_ratio * 10)
        
        print(f"📝 **Text Quality:**")
        print(f"   Total characters: {total_chars}")
        print(f"   Garbled chars (�): {garbled_chars}")
        print(f"   Weird characters: {weird_chars}")
        print(f"   Problem ratio: {problem_ratio:.1%}")
        print(f"   Extractability: {extractability:.1%}")
        
        # 4. Overall Scoring
        score = 0.0
        reasoning = []
        
        # Math scoring
        if math_density > 0.1:
            score += 0.4
            reasoning.append(f"High math density ({math_density:.1%})")
        elif math_density > 0.05:
            score += 0.2
            reasoning.append(f"Moderate math content ({math_density:.1%})")
        
        # Layout scoring  
        if layout_complexity > 0.3:
            score += 0.3
            reasoning.append(f"Complex layout detected ({layout_complexity:.1%})")
        elif layout_complexity > 0.1:
            score += 0.1
            reasoning.append(f"Some layout complexity ({layout_complexity:.1%})")
        
        # Extractability scoring
        if extractability < 0.5:
            score += 0.5
            reasoning.append(f"Poor text extraction ({extractability:.1%})")
        elif extractability < 0.8:
            score += 0.2
            reasoning.append(f"Moderate extraction issues ({extractability:.1%})")
        
        # Tier determination
        if score >= 0.6:
            tier = "PREMIUM"
        elif score >= 0.3:
            tier = "SMART"  
        else:
            tier = "FAST"
        
        print(f"🎯 **Final Assessment:**")
        print(f"   Total score: {score:.2f}")
        print(f"   Recommended tier: {tier}")
        print(f"   Reasoning: {', '.join(reasoning) if reasoning else 'Simple document'}")
    
    print("\n" + "="*80)
    print("HOW IT WORKS:")
    print("="*80)
    print("""
🔍 **Text Extraction**: First 3 pages extracted with pdfplumber
📊 **Pattern Matching**: Python regex (re.findall) searches for specific patterns
🧮 **Math Detection**: Counts LaTeX, symbols, Greek letters per word
📐 **Layout Analysis**: Finds figures/tables + analyzes line lengths for multi-column
📝 **Quality Check**: Counts garbled characters and extraction artifacts
🎯 **Scoring**: Adds weighted scores for each factor (0.0-1.0+)
⚡ **Tier Selection**: Score thresholds determine FAST/SMART/PREMIUM

**Key Insight**: The system analyzes *extracted text*, not visual layout!
So it detects LaTeX source code, Unicode symbols, and text patterns
rather than actual rendered equations or visual elements.
""")

if __name__ == "__main__":
    demo_pattern_detection()