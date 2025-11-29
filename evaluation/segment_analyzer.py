"""
Segment-level analysis utilities.
Breaks down text into segments and provides token-level explainability.
"""

import re
from typing import List, Dict, Any


class SegmentAnalyzer:
    """Utilities for analyzing segments of text."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        # Split on . ! ? followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def count_hallucinated_tokens(
        segment_labels: List[Dict[str, Any]],
        tokenizer_func
    ) -> int:
        """
        Count tokens in hallucinated segments.
        
        Args:
            segment_labels: List of segment label dicts
            tokenizer_func: Function to count tokens in text
        
        Returns:
            Total hallucinated tokens
        """
        total = 0
        for segment in segment_labels:
            if segment.get("is_hallucination", False):
                total += tokenizer_func(segment["segment_text"])
        return total
    
    @staticmethod
    def get_segments_by_label(
        segment_labels: List[Dict[str, Any]],
        label: str
    ) -> List[Dict[str, Any]]:
        """
        Filter segments by label.
        
        Args:
            segment_labels: List of segment label dicts
            label: Label to filter by
        
        Returns:
            Filtered segments
        """
        return [seg for seg in segment_labels if seg.get("label") == label]
    
    @staticmethod
    def generate_highlighted_text(
        original_text: str,
        segment_labels: List[Dict[str, Any]]
    ) -> str:
        """
        Generate HTML with highlighted segments.
        
        Args:
            original_text: Original text
            segment_labels: Segment labels with flags
        
        Returns:
            HTML string with color-coded segments
        """
        html_parts = []
        
        for segment in segment_labels:
            text = segment["segment_text"]
            
            # Determine color based on flags
            if segment.get("is_safety_violation"):
                color = "#ff0000"  # Red
                label = "SAFETY VIOLATION"
            elif segment.get("is_hallucination"):
                color = "#ff6b6b"  # Light red
                label = "HALLUCINATION"
            elif segment.get("is_potential_fake_news"):
                color = "#ffa500"  # Orange
                label = "FAKE NEWS"
            elif segment.get("is_wrong_answer"):
                color = "#ffeb3b"  # Yellow
                label = "WRONG"
            else:
                color = "#4caf50"  # Green
                label = "OK"
            
            html_parts.append(
                f'<span style="background-color: {color}; padding: 2px 4px; '
                f'border-radius: 3px; margin: 2px;" title="{label}">{text}</span>'
            )
        
        return " ".join(html_parts)
    
    @staticmethod
    def get_segment_statistics(segment_labels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute statistics about segments.
        
        Args:
            segment_labels: List of segment label dicts
        
        Returns:
            Statistics dict
        """
        total = len(segment_labels)
        if total == 0:
            return {
                "total_segments": 0,
                "hallucination_segments": 0,
                "safety_violation_segments": 0,
                "fake_news_segments": 0,
                "wrong_answer_segments": 0,
                "correct_segments": 0
            }
        
        hallucination = sum(1 for s in segment_labels if s.get("is_hallucination"))
        safety = sum(1 for s in segment_labels if s.get("is_safety_violation"))
        fake_news = sum(1 for s in segment_labels if s.get("is_potential_fake_news"))
        wrong = sum(1 for s in segment_labels if s.get("is_wrong_answer"))
        correct = sum(
            1 for s in segment_labels
            if not any([
                s.get("is_hallucination"),
                s.get("is_safety_violation"),
                s.get("is_potential_fake_news"),
                s.get("is_wrong_answer")
            ])
        )
        
        return {
            "total_segments": total,
            "hallucination_segments": hallucination,
            "hallucination_pct": (hallucination / total) * 100,
            "safety_violation_segments": safety,
            "safety_violation_pct": (safety / total) * 100,
            "fake_news_segments": fake_news,
            "fake_news_pct": (fake_news / total) * 100,
            "wrong_answer_segments": wrong,
            "wrong_answer_pct": (wrong / total) * 100,
            "correct_segments": correct,
            "correct_pct": (correct / total) * 100
        }
