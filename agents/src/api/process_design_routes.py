#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Design API Routes.
This module defines the FastAPI routes for process design and analysis.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Body, Query, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from pydantic import BaseModel, Field
import uuid
import time
import os
import numpy as np

from src.agents.agent_framework import AgentFramework, AgentRole, Agent
from src.process_engine.gnn_model import GNNModel
from src.vector_db.vector_store import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/process",
    tags=["Process Design"],
    responses={404: {"description": "Not found"}},
)

# Models for API requests and responses
class ProcessRequirements(BaseModel):
    """Process requirements for design."""
    process_name: str
    domain: str = "general"
    description: Optional[str] = None
    requirements: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    expected_activities: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    
class ProcessAnalysisRequest(BaseModel):
    """Request for process analysis."""
    process_id: str
    focus_areas: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ProcessSimilarityRequest(BaseModel):
    """Request for finding similar processes."""
    process_id: str
    num_results: int = 5
    filters: Dict[str, Any] = Field(default_factory=dict)

class ProcessTaskResponse(BaseModel):
    """Response containing task information."""
    task_id: str
    status: str
    message: str

class ProcessMetadata(BaseModel):
    """Metadata for a process."""
    name: str = Field(..., description="Name of the process")
    description: Optional[str] = Field(None, description="Description of the process")
    domain: Optional[str] = Field(None, description="Domain of the process")
    version: Optional[str] = Field(None, description="Version of the process")
    tags: Optional[List[str]] = Field(None, description="Tags for the process")
    author: Optional[str] = Field(None, description="Author of the process")
    created_by: Optional[str] = Field(None, description="User who created the process")
    
class ProcessNode(BaseModel):
    """Node in a process graph."""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type of the node (task, gateway, event, etc.)")
    name: str = Field(..., description="Name of the node")
    description: Optional[str] = Field(None, description="Description of the node")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    position: Optional[Dict[str, float]] = Field(None, description="Position for rendering")

class ProcessEdge(BaseModel):
    """Edge in a process graph."""
    id: str = Field(..., description="Unique identifier for the edge")
    source: str = Field(..., description="ID of the source node")
    target: str = Field(..., description="ID of the target node")
    type: str = Field(..., description="Type of the edge (sequence, message, etc.)")
    name: Optional[str] = Field(None, description="Name of the edge")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")

class ProcessGraph(BaseModel):
    """Process graph representation."""
    nodes: List[ProcessNode] = Field(..., description="Nodes in the process")
    edges: List[ProcessEdge] = Field(..., description="Edges in the process")

class Process(BaseModel):
    """Complete process model."""
    metadata: ProcessMetadata = Field(..., description="Process metadata")
    graph: ProcessGraph = Field(..., description="Process graph")
    
class ProcessAnalysisResult(BaseModel):
    """Result of process analysis."""
    process_id: Optional[str] = Field(None, description="ID of the analyzed process")
    embedding: Optional[List[float]] = Field(None, description="Process embedding vector")
    complexity: Optional[float] = Field(None, description="Process complexity score")
    bottlenecks: Optional[List[Dict[str, Any]]] = Field(None, description="Identified bottlenecks")
    optimization_opportunities: Optional[List[Dict[str, Any]]] = Field(None, description="Optimization opportunities")
    similar_processes: Optional[List[Dict[str, Any]]] = Field(None, description="Similar processes")
    
class ProcessDesignRequest(BaseModel):
    """Request for process design."""
    name: str = Field(..., description="Name for the new process")
    description: Optional[str] = Field(None, description="Description of the process to design")
    domain: Optional[str] = Field(None, description="Domain for the process")
    requirements: Optional[List[str]] = Field(None, description="Requirements for the process")
    similar_to_process_id: Optional[str] = Field(None, description="ID of a similar process to use as reference")
    
class ProcessRecommendationRequest(BaseModel):
    """Request for process improvement recommendations."""
    process_id: str = Field(..., description="ID of the process to get recommendations for")
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus recommendations on")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints for recommendations")

class ProcessComparisonRequest(BaseModel):
    """Request for process comparison."""
    process_id_1: str = Field(..., description="ID of the first process")
    process_id_2: str = Field(..., description="ID of the second process")
    comparison_aspects: Optional[List[str]] = Field(None, description="Aspects to compare")

class ProcessListResponse(BaseModel):
    """Response model for process list."""
    process_id: str
    name: str
    description: str
    domain: str
    complexity: float
    upload_time: str

# Define dependencies
async def get_agent_framework():
    """Get the agent framework from the FastAPI app state."""
    from src.main import app
    return app.state._agent_framework

async def get_gnn_model():
    """Get the GNN model from the app state."""
    from src.main import _gnn_model
    return _gnn_model

async def get_vector_store():
    """Get the vector store from the app state."""
    from src.main import _vector_store
    return _vector_store

# Routes for process design
@router.post("/design", response_model=ProcessTaskResponse)
async def design_process(
    requirements: ProcessRequirements,
    agent_framework: AgentFramework = Depends(get_agent_framework)
):
    """
    Start a new process design task.
    
    Args:
        requirements: Process requirements
        agent_framework: Agent framework dependency
        
    Returns:
        Task information
    """
    # Get the coordinator agent
    coordinator = agent_framework.get_agent_by_role(AgentRole.COORDINATOR)
    
    if not coordinator:
        raise HTTPException(status_code=500, detail="Coordinator agent not found")
    
    # Create a unique task ID
    task_id = str(uuid.uuid4())
    
    # Send design request to coordinator
    content = {
        "workflow_type": "process_design",
        "process_data": requirements.dict()
    }
    
    await coordinator.send_message(
        receiver=coordinator.agent_id,
        content=content,
        message_type="start_workflow",
        correlation_id=task_id
    )
    
    # Return task information
    return ProcessTaskResponse(
        task_id=task_id,
        status="started",
        message="Process design task started"
    )

@router.post("/upload", response_model=Dict[str, Any])
async def upload_process(
    file: UploadFile = File(...),
    gnn_model: GNNModel = Depends(get_gnn_model),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Upload a process model file.
    
    Args:
        file: Process model file
        gnn_model: GNN model
        vector_store: Vector store
        
    Returns:
        Process information with assigned ID
    """
    try:
        # Read file contents
        contents = await file.read()
        
        # Parse process file
        process_data = gnn_model.parse_process_file(contents, file.filename)
        
        # Convert to graph
        process_graph = gnn_model.convert_to_graph(process_data)
        
        # Generate embedding
        embedding = gnn_model.generate_embedding(process_graph)
        
        # Add to vector store
        metadata = {
            "name": process_data.get("metadata", {}).get("name", "Unnamed Process"),
            "format": process_data.get("metadata", {}).get("format", "unknown"),
            "original_filename": file.filename,
            "node_count": len(process_data.get("nodes", [])),
            "edge_count": len(process_data.get("edges", [])),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        vector_id = vector_store.add(embedding, metadata)
        
        return {
            "id": vector_id,
            "name": metadata["name"],
            "format": metadata["format"],
            "node_count": metadata["node_count"],
            "edge_count": metadata["edge_count"],
            "message": "Process uploaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading process: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_process(
    request: ProcessAnalysisRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    gnn_model: GNNModel = Depends(get_gnn_model)
):
    """
    Analyze a process and return insights.
    """
    # Check if process exists in vector store
    process_metadata = vector_store.get(request.process_id)
    
    if not process_metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Process with ID {request.process_id} not found"
        )
    
    # Get process data from metadata
    process_data = process_metadata.get('process_design', {})
    
    # Analyze process using GNN model
    analysis_result = gnn_model.analyze_process(process_data)
    
    # Add process ID to result
    analysis_result['process_id'] = request.process_id
    
    return analysis_result

@router.post("/design", response_model=Process, status_code=status.HTTP_201_CREATED)
async def design_process(
    request: ProcessDesignRequest,
    agent_framework = Depends(get_agent_framework),
    vector_store = Depends(get_vector_store),
    gnn_model = Depends(get_gnn_model)
):
    """
    Design a new process based on requirements.
    
    This endpoint leverages the agent framework to create a new process design
    based on the provided requirements and domain knowledge.
    """
    try:
        # Prepare request for agent framework
        design_request = {
            "name": request.name,
            "description": request.description,
            "domain": request.domain,
            "requirements": request.requirements
        }
        
        # If we have a reference process, add its embedding
        if request.similar_to_process_id:
            process_metadata = vector_store.get(request.similar_to_process_id)
            if process_metadata and "embedding" in process_metadata:
                design_request["reference_embedding"] = process_metadata["embedding"]
        
        # Process the request through the agent framework
        result = await agent_framework.process_request(
            "process_design", 
            design_request
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Process design should be in the results
        process_design = result.get("results", {}).get("process_design")
        if not process_design:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="No process design generated"
            )
        
        # Store the process in the vector store
        if "graph" in process_design:
            # Generate embedding for the process
            embedding = gnn_model.get_process_embedding(process_design)
            
            # Store in vector database
            process_id = vector_store.add(
                embedding=embedding,
                metadata={
                    "name": request.name,
                    "description": request.description,
                    "domain": request.domain,
                    "process_definition": process_design
                }
            )
            
            # Save the vector store
            vector_store.save()
            
            # Add process ID to the result
            process_design["metadata"]["id"] = process_id
        
        return process_design
        
    except Exception as e:
        logger.error(f"Error designing process: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error designing process: {str(e)}"
        )

@router.get("/list", response_model=List[ProcessListResponse])
async def list_processes(
    vector_store: VectorStore = Depends(get_vector_store),
    agent_framework: AgentFramework = Depends(get_agent_framework),
    limit: int = Query(20, description="Maximum number of processes to return"),
    offset: int = Query(0, description="Number of processes to skip")
):
    """
    List available processes.
    
    Args:
        vector_store: Vector store dependency
        agent_framework: Agent framework dependency
        limit: Maximum number of processes to return
        offset: Number of processes to skip
        
    Returns:
        List of processes
    """
    # Get coordinator agent
    coordinator = agent_framework.get_agent_by_role(AgentRole.COORDINATOR)
    processes = []
    
    # Debug information
    logger.debug(f"Listing processes, coordinator: {coordinator is not None}")
    if coordinator:
        logger.debug(f"Active workflows: {len(coordinator.active_workflows)}")
        for wf_id, workflow in coordinator.active_workflows.items():
            logger.debug(f"Workflow {wf_id}: status={workflow['status']}, results={list(workflow['results'].keys())}")
    
    # Collect processes from completed workflows
    if coordinator:
        for workflow_id, workflow in coordinator.active_workflows.items():
            if workflow["status"] == "completed" and "process_design" in workflow["results"]:
                process_design = workflow["results"]["process_design"]
                if process_design:
                    metadata = process_design.get("metadata", {})
                    process_id = workflow["results"].get("process_id", workflow_id)
                    
                    # Check if process exists in vector store, if not, add it
                    if vector_store.get(process_id) is None:
                        # Create a simple embedding (placeholder)
                        dummy_embedding = np.zeros(vector_store.dimension, dtype=np.float32)
                        
                        # Add process metadata to vector store
                        vector_store.add(
                            embedding=dummy_embedding,
                            metadata={
                                "name": metadata.get("name", "Unnamed Process"),
                                "description": metadata.get("description", ""),
                                "domain": metadata.get("domain", "general"),
                                "complexity": 0.5,  # Default complexity
                                "upload_time": workflow.get("end_time", ""),
                                "process_design": process_design
                            },
                            process_id=process_id
                        )
                        
                        # Save the vector store to persist the data
                        vector_store.save()
                    
                    processes.append(ProcessListResponse(
                        process_id=process_id,
                        name=metadata.get("name", "Unnamed Process"),
                        description=metadata.get("description", ""),
                        domain=metadata.get("domain", "general"),
                        complexity=0.5,  # Default complexity
                        upload_time=workflow.get("end_time", "")
                    ))
    
    # Also get processes from vector store
    for process_id, metadata in vector_store.metadata.items():
        # Skip if already added from workflows
        if any(p.process_id == process_id for p in processes):
            continue
            
        processes.append(ProcessListResponse(
            process_id=process_id,
            name=metadata.get("name", "Unnamed Process"),
            description=metadata.get("description", ""),
            domain=metadata.get("domain", "general"),
            complexity=metadata.get("complexity", 0.5),
            upload_time=metadata.get("upload_time", "")
        ))
    
    # Apply limit and offset
    return processes[offset:offset+limit]

@router.get("/{process_id}", response_model=Dict[str, Any])
async def get_process(
    process_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Get a process by ID.
    
    Args:
        process_id: Process ID
        vector_store: Vector store
        
    Returns:
        Process metadata
    """
    try:
        # Get process metadata from vector store
        process_metadata = vector_store.get(process_id)
        
        if not process_metadata:
            raise HTTPException(status_code=404, detail=f"Process with ID {process_id} not found")
        
        return {
            "id": process_id,
            **process_metadata
        }
    
    except Exception as e:
        logger.error(f"Error getting process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting process: {str(e)}")

@router.delete("/{process_id}", response_model=Dict[str, Any])
async def delete_process(
    process_id: str,
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    Delete a process by ID.
    
    Args:
        process_id: Process ID
        vector_store: Vector store
        
    Returns:
        Status message
    """
    try:
        # Delete process
        process_id = int(process_id)
        if vector_store.delete(process_id):
            return {"message": f"Process with ID {process_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Process with ID {process_id} not found")
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid process ID")
    except Exception as e:
        logger.error(f"Error deleting process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting process: {str(e)}")

@router.get("/task/{task_id}", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str,
    agent_framework: AgentFramework = Depends(get_agent_framework)
):
    """
    Get the status of a task.
    
    Args:
        task_id: Task ID
        agent_framework: Agent framework dependency
        
    Returns:
        Task status
    """
    # Get coordinator agent
    coordinator = agent_framework.get_agent_by_role(AgentRole.COORDINATOR)
    if not coordinator:
        raise HTTPException(status_code=500, detail="Coordinator agent not found")
    
    # Check if workflow exists
    workflow = coordinator.active_workflows.get(task_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    # Calculate progress based on completed tasks
    total_tasks = len(workflow["pending_tasks"]) + len(workflow["completed_tasks"])
    progress = len(workflow["completed_tasks"]) / total_tasks if total_tasks > 0 else 0
    
    return {
        "task_id": task_id,
        "status": workflow["status"],
        "progress": progress,
        "message": f"Task is {workflow['status']}",
        "created_at": workflow["start_time"]
    }

@router.post("/recommend", status_code=status.HTTP_200_OK)
async def recommend_improvements(
    request: ProcessRecommendationRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    gnn_model: GNNModel = Depends(get_gnn_model)
):
    """
    Generate recommendations for improving a process.
    """
    try:
        # Check if process exists
        process_metadata = vector_store.get(request.process_id)
        if not process_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process with ID {request.process_id} not found"
            )
        
        # Get process data
        process_data = process_metadata.get('process_design', {})
        
        # Extract metadata
        metadata = process_data.get("metadata", {})
        process_name = metadata.get("name", "Unnamed Process")
        
        # Generate mock recommendations
        return {
            "process_id": request.process_id,
            "process_name": process_name,
            "recommendations": [
                {
                    "type": "efficiency",
                    "description": "Consider implementing parallel processing for document collection and background check activities.",
                    "impact": "high",
                    "effort": "medium",
                    "roi": 0.8
                },
                {
                    "type": "automation",
                    "description": "Automate the IT setup process using predefined templates and scripts.",
                    "impact": "medium",
                    "effort": "low",
                    "roi": 0.9
                },
                {
                    "type": "quality",
                    "description": "Add a feedback loop after training to ensure knowledge retention.",
                    "impact": "medium",
                    "effort": "low",
                    "roi": 0.7
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )

@router.post("/compare", status_code=status.HTTP_200_OK)
async def compare_processes(
    request: ProcessComparisonRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    gnn_model: GNNModel = Depends(get_gnn_model)
):
    """
    Compare two processes.
    """
    try:
        # Check if both processes exist
        process1_metadata = vector_store.get(request.process_id_1)
        if not process1_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process with ID {request.process_id_1} not found"
            )
        
        process2_metadata = vector_store.get(request.process_id_2)
        if not process2_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process with ID {request.process_id_2} not found"
            )
        
        # Get process data
        process1_data = process1_metadata.get('process_design', {})
        process2_data = process2_metadata.get('process_design', {})
        
        # Extract metadata
        metadata1 = process1_data.get("metadata", {})
        metadata2 = process2_data.get("metadata", {})
        
        process1_name = metadata1.get("name", "Unnamed Process 1")
        process2_name = metadata2.get("name", "Unnamed Process 2")
        
        # Extract nodes and edges
        nodes1 = process1_data.get("graph", {}).get("nodes", [])
        edges1 = process1_data.get("graph", {}).get("edges", [])
        
        nodes2 = process2_data.get("graph", {}).get("nodes", [])
        edges2 = process2_data.get("graph", {}).get("edges", [])
        
        # Generate mock comparison
        return {
            "processes": [
                {
                    "process_id": request.process_id_1,
                    "name": process1_name,
                    "node_count": len(nodes1),
                    "edge_count": len(edges1)
                },
                {
                    "process_id": request.process_id_2,
                    "name": process2_name,
                    "node_count": len(nodes2),
                    "edge_count": len(edges2)
                }
            ],
            "comparison": {
                "structural_similarity": 0.65,
                "semantic_similarity": 0.72,
                "performance_comparison": {
                    "efficiency": 0.8 if process1_name == "HR Onboarding Process" else 0.6,
                    "quality": 0.7 if process1_name == "HR Onboarding Process" else 0.8,
                    "compliance": 0.9
                },
                "differences": [
                    {
                        "type": "structure",
                        "description": f"{process1_name} has {len(nodes1)} activities while {process2_name} has {len(nodes2)} activities."
                    },
                    {
                        "type": "flow",
                        "description": "The processes have different control flow patterns."
                    }
                ],
                "common_elements": [
                    {
                        "type": "activity",
                        "description": "Both processes have start and end activities."
                    }
                ]
            }
        }
    except Exception as e:
        logger.error(f"Error comparing processes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing processes: {str(e)}"
        )

@router.post("/similar", status_code=status.HTTP_200_OK)
async def find_similar_processes(
    request: ProcessSimilarityRequest,
    vector_store = Depends(get_vector_store),
    gnn_model = Depends(get_gnn_model)
):
    """
    Find processes similar to a given process.
    """
    try:
        # Check if process exists
        process_metadata = vector_store.get(request.process_id)
        
        if not process_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Process with ID {request.process_id} not found"
            )
        
        # Get process embedding
        process_data = process_metadata.get('process_data', {})
        analysis = gnn_model.analyze_process(process_data)
        embedding = analysis.get('embedding', [])
        
        # Apply filters if provided
        filtered_ids = None
        if request.filters:
            filtered_ids = set(vector_store.search_by_metadata(request.filters))
        
        # Find similar processes
        similar = vector_store.search(
            query_embedding=embedding,
            k=request.num_results + 1,  # Add one to account for self
            return_distances=True
        )
        
        # Process results
        results = []
        for process_id, distance in similar:
            if process_id != request.process_id:  # Skip self
                if filtered_ids is None or process_id in filtered_ids:
                    process_meta = vector_store.get(process_id)
                    if process_meta:
                        results.append({
                            'process_id': process_id,
                            'name': process_meta.get('name', 'Unknown'),
                            'description': process_meta.get('description', ''),
                            'similarity': 1.0 - min(1.0, distance / 10.0),  # Convert distance to similarity score
                            'domain': process_meta.get('domain', 'general')
                        })
        
        return {
            'process_id': request.process_id,
            'similar_processes': results
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error finding similar processes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding similar processes: {str(e)}"
        ) 