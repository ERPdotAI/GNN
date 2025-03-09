#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the AgenticProcessGNN application.
This module initializes all components and starts the API server.
"""

import os
import logging
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AgenticProcessGNN",
    description="A multi-agent system leveraging Graph Neural Networks for process design, analysis, and optimization",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import components
from src.process_engine.gnn_model import GNNModel
from src.vector_db.vector_store import VectorStore
from src.agents.agent_framework import AgentFramework

# Global variables to store instances
_vector_store: Optional[VectorStore] = None
_gnn_model: Optional[GNNModel] = None
_agent_framework: Optional[AgentFramework] = None

# Import API routers - moved after app creation to avoid circular imports
from src.api.process_design_routes import router as process_router
from src.api.integration_routes import router as integration_router

# Include API routers
app.include_router(process_router, prefix="/process", tags=["Process Design"])
app.include_router(integration_router, prefix="/integration", tags=["Integration"])

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    global _vector_store, _gnn_model, _agent_framework
    
    logger.info("Initializing Vector Store...")
    vector_store_dir = os.path.join("data", "vector_store")
    os.makedirs(vector_store_dir, exist_ok=True)
    _vector_store = VectorStore(
        dimension=512,
        index_type="Flat",
        storage_dir=vector_store_dir
    )
    
    logger.info("Initializing GNN Model...")
    model_path = os.path.join("models", "gnn_model.pt")
    _gnn_model = GNNModel(model_path=model_path if os.path.exists(model_path) else None)
    
    logger.info("Initializing Agent Framework...")
    _agent_framework = AgentFramework(
        gnn_model=_gnn_model,
        vector_store=_vector_store
    )
    
    # Store in app.state for dependency injection
    app.state._agent_framework = _agent_framework
    
    logger.info("Starting agents...")
    await _agent_framework.start_agents()
    
    logger.info("Startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    global _vector_store, _agent_framework
    
    logger.info("Stopping agents...")
    if _agent_framework:
        await _agent_framework.stop_agents()
    
    logger.info("Saving vector store...")
    if _vector_store:
        _vector_store.save()
    
    logger.info("Shutdown complete.")

@app.get("/")
async def root():
    """Root endpoint returning system information."""
    return {
        "name": "AgenticProcessGNN",
        "status": "online",
        "version": "0.1.0",
        "components": {
            "gnn_model": "initialized" if _gnn_model else "not initialized",
            "vector_store": "initialized" if _vector_store else "not initialized",
            "agent_framework": "initialized" if _agent_framework else "not initialized",
        }
    }

def main():
    """Main entry point."""
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    os.makedirs(os.path.join("data", "reference_processes"), exist_ok=True)

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        workers=1,    # Use single worker for agent framework
    )

if __name__ == "__main__":
    main() 