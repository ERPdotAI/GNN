"""
Process-related API routes for the AgenticProcessGNN system.
These endpoints handle process upload, analysis, and retrieval.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Import application components
from process_engine.gnn_model import ProcessGNN
from vector_db.process_db import ProcessVectorDB
from agents.coordinator import AgentCoordinator

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

# Data models
class ProcessAnalysisRequest(BaseModel):
    """Request model for process analysis."""
    process_data: Dict[str, Any]
    analysis_type: str = "full"  # Options: full, structure, compliance, efficiency, risk

class ProcessQueryRequest(BaseModel):
    """Request model for querying similar processes."""
    description: str
    domain: Optional[str] = None
    constraints: Optional[List[str]] = None
    limit: int = 5

class ProcessRecommendationRequest(BaseModel):
    """Request model for process improvement recommendations."""
    process_id: str
    focus_areas: Optional[List[str]] = None

# Dependencies
def get_process_gnn():
    """Dependency to get the GNN model."""
    from main import process_gnn
    if not process_gnn:
        raise HTTPException(status_code=503, detail="Process GNN model not initialized")
    return process_gnn

def get_vector_db():
    """Dependency to get the vector database."""
    from main import vector_db
    if not vector_db:
        raise HTTPException(status_code=503, detail="Vector database not initialized")
    return vector_db

def get_agent_coordinator():
    """Dependency to get the agent coordinator."""
    from main import agent_coordinator
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")
    return agent_coordinator

# Routes
@router.post("/analyze")
async def analyze_process(
    request: ProcessAnalysisRequest,
    process_gnn: ProcessGNN = Depends(get_process_gnn),
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Analyze a process and return insights."""
    try:
        logger.info(f"Analyzing process with type: {request.analysis_type}")
        
        # Convert process data to graph representation
        process_graph = process_gnn.convert_to_graph(request.process_data)
        
        # Analyze the process with appropriate agents
        analysis_results = await agent_coordinator.analyze_process(
            process_graph, 
            analysis_type=request.analysis_type
        )
        
        return analysis_results
    except Exception as e:
        logger.error(f"Error analyzing process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing process: {str(e)}")

@router.post("/upload")
async def upload_process(
    file: UploadFile = File(...),
    vector_db: ProcessVectorDB = Depends(get_vector_db),
    process_gnn: ProcessGNN = Depends(get_process_gnn)
):
    """Upload and store a process model."""
    try:
        logger.info(f"Uploading process file: {file.filename}")
        
        # Read and parse the uploaded file
        contents = await file.read()
        process_data = process_gnn.parse_process_file(contents, file.filename)
        
        # Convert to graph and generate embedding
        process_graph = process_gnn.convert_to_graph(process_data)
        embedding = process_gnn.generate_embedding(process_graph)
        
        # Store in vector database
        process_id = await vector_db.store_process(
            process_data=process_data,
            embedding=embedding.tolist(),
            metadata={
                "filename": file.filename,
                "file_type": file.content_type
            }
        )
        
        return {"process_id": process_id, "message": "Process uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading process: {str(e)}")

@router.post("/search")
async def search_similar_processes(
    query: ProcessQueryRequest,
    vector_db: ProcessVectorDB = Depends(get_vector_db),
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Search for similar processes based on description."""
    try:
        logger.info(f"Searching processes with query: {query.description}")
        
        # Generate embedding for the query
        query_embedding = await agent_coordinator.generate_query_embedding(
            query.description,
            domain=query.domain,
            constraints=query.constraints
        )
        
        # Search vector database
        similar_processes = await vector_db.search_processes(
            embedding=query_embedding,
            limit=query.limit,
            filter_criteria={
                "domain": query.domain
            } if query.domain else None
        )
        
        return similar_processes
    except Exception as e:
        logger.error(f"Error searching processes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching processes: {str(e)}")

@router.post("/recommend")
async def get_process_recommendations(
    request: ProcessRecommendationRequest,
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator),
    vector_db: ProcessVectorDB = Depends(get_vector_db)
):
    """Get recommendations for improving a process."""
    try:
        logger.info(f"Getting recommendations for process: {request.process_id}")
        
        # Retrieve process from database
        process_data = await vector_db.get_process(request.process_id)
        if not process_data:
            raise HTTPException(status_code=404, detail="Process not found")
        
        # Generate recommendations
        recommendations = await agent_coordinator.generate_recommendations(
            process_data=process_data,
            focus_areas=request.focus_areas
        )
        
        return recommendations
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/{process_id}")
async def get_process(
    process_id: str,
    vector_db: ProcessVectorDB = Depends(get_vector_db)
):
    """Retrieve a specific process by ID."""
    try:
        logger.info(f"Retrieving process: {process_id}")
        
        process = await vector_db.get_process(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Process not found")
            
        return process
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving process: {str(e)}") 