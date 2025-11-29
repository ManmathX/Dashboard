"""
FastAPI main application.
Entry point for the LLM Evaluation Framework API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config import settings
from api.routes import router
from models.database import db


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting LLM Evaluation Framework...")
    
    # Validate API keys
    try:
        settings.validate_api_keys()
        logger.info("API keys validated successfully")
    except ValueError as e:
        logger.warning(f"API key validation warning: {e}")
    
    # Connect to database
    try:
        await db.connect()
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await db.disconnect()
    logger.info("Disconnected from MongoDB")


# Create FastAPI app
app = FastAPI(
    title="LLM Evaluation Framework",
    description="Multi-LLM evaluation and scoring system with hallucination detection, "
                "jailbreak detection, fake news detection, and correctness scoring",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins including null
    allow_credentials=False,  # Must be False when using wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["evaluation"])


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files at the root
# This must be after API routes to avoid conflicts
app.mount("/", StaticFiles(directory="dashboard", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
