"""
Database models for MongoDB storage.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

from config import settings


class Database:
    """MongoDB database interface."""
    
    def __init__(self):
        """Initialize database connection."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    async def connect(self):
        """Connect to MongoDB."""
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_database]
        
        # Create indexes
        await self._create_indexes()
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
    
    async def _create_indexes(self):
        """Create database indexes for performance."""
        # Evaluations collection
        evaluations = self.db.evaluations
        await evaluations.create_index([("evaluation_id", ASCENDING)], unique=True)
        await evaluations.create_index([("timestamp", DESCENDING)])
        await evaluations.create_index([("input_data.prompt_id", ASCENDING)])
        
        # Metrics collection
        metrics = self.db.dataset_metrics
        await metrics.create_index([("timestamp", DESCENDING)])
    
    async def save_evaluation(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Save evaluation result to database.
        
        Args:
            evaluation_result: Evaluation result dict
        
        Returns:
            Evaluation ID
        """
        result = await self.db.evaluations.insert_one(evaluation_result)
        return str(result.inserted_id)
    
    async def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve evaluation by ID.
        
        Args:
            evaluation_id: Evaluation ID
        
        Returns:
            Evaluation result or None
        """
        return await self.db.evaluations.find_one({"evaluation_id": evaluation_id})
    
    async def get_evaluations(
        self,
        limit: int = 100,
        skip: int = 0,
        sort_by: str = "timestamp",
        sort_order: int = -1
    ) -> list:
        """
        Get multiple evaluations.
        
        Args:
            limit: Maximum number of results
            skip: Number of results to skip
            sort_by: Field to sort by
            sort_order: 1 for ascending, -1 for descending
        
        Returns:
            List of evaluations
        """
        cursor = self.db.evaluations.find().sort(sort_by, sort_order).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def save_dataset_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Save dataset-level metrics.
        
        Args:
            metrics: Metrics dict
        
        Returns:
            Metrics ID
        """
        metrics["timestamp"] = datetime.utcnow()
        result = await self.db.dataset_metrics.insert_one(metrics)
        return str(result.inserted_id)
    
    async def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get latest dataset metrics.
        
        Returns:
            Latest metrics or None
        """
        return await self.db.dataset_metrics.find_one(sort=[("timestamp", DESCENDING)])


# Global database instance
db = Database()
