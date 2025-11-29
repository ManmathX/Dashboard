"""
Database models for Mock storage (No-DB mode).
"""

from datetime import datetime
from typing import Optional, Dict, Any
import uuid

class Database:
    """Mock database interface that does not store data persistently."""
    
    def __init__(self):
        """Initialize database connection."""
        pass
    
    async def connect(self):
        """Mock connect."""
        pass
    
    async def disconnect(self):
        """Mock disconnect."""
        pass
    
    async def save_evaluation(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Mock save evaluation.
        Returns a fake ID.
        """
        return str(uuid.uuid4())
    
    async def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Mock get evaluation."""
        return None
    
    async def get_evaluations(
        self,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "timestamp",
        sort_order: int = -1
    ) -> list:
        """Mock get evaluations."""
        return []
    
    async def save_dataset_metrics(self, metrics: Dict[str, Any]) -> str:
        """Mock save metrics."""
        return str(uuid.uuid4())
    
    async def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Mock get latest metrics."""
        return {
            "avg_hallucination_pct": 0.0,
            "avg_jailbreak_pct": 0.0,
            "avg_fake_news_pct": 0.0,
            "total_evaluations": 0
        }


# Global database instance
db = Database()
