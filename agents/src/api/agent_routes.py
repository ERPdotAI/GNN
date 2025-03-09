"""
Agent-related API routes for the AgenticProcessGNN system.
These endpoints handle agent interactions and operations.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

# Import application components
from agents.coordinator import AgentCoordinator

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

# Data models
class AgentQuery(BaseModel):
    """Request model for agent queries."""
    query: str
    context: Optional[Dict[str, Any]] = None
    agent_type: Optional[str] = None  # If None, query goes to coordinator

class ProcessDesignRequest(BaseModel):
    """Request model for process design."""
    description: str
    domain: Optional[str] = None
    requirements: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    base_process_id: Optional[str] = None

class ProcessOptimizationRequest(BaseModel):
    """Request model for process optimization."""
    process_id: str
    optimization_goals: List[str]
    constraints: Optional[List[str]] = None

# Dependencies
def get_agent_coordinator():
    """Dependency to get the agent coordinator."""
    from main import agent_coordinator
    if not agent_coordinator:
        raise HTTPException(status_code=503, detail="Agent coordinator not initialized")
    return agent_coordinator

# Routes
@router.post("/query")
async def query_agents(
    query: AgentQuery,
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Send a query to the agent system."""
    try:
        logger.info(f"Processing agent query: {query.query[:50]}...")
        
        response = await agent_coordinator.process_query(
            query=query.query,
            context=query.context,
            agent_type=query.agent_type
        )
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing agent query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing agent query: {str(e)}")

@router.post("/design")
async def design_process(
    request: ProcessDesignRequest,
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Design a new process based on requirements."""
    try:
        logger.info(f"Designing process for: {request.description[:50]}...")
        
        process_design = await agent_coordinator.design_process(
            description=request.description,
            domain=request.domain,
            requirements=request.requirements,
            constraints=request.constraints,
            base_process_id=request.base_process_id
        )
        
        return process_design
    except Exception as e:
        logger.error(f"Error designing process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error designing process: {str(e)}")

@router.post("/optimize")
async def optimize_process(
    request: ProcessOptimizationRequest,
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Optimize an existing process."""
    try:
        logger.info(f"Optimizing process: {request.process_id}")
        
        optimized_process = await agent_coordinator.optimize_process(
            process_id=request.process_id,
            optimization_goals=request.optimization_goals,
            constraints=request.constraints
        )
        
        return optimized_process
    except Exception as e:
        logger.error(f"Error optimizing process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error optimizing process: {str(e)}")

@router.get("/types")
async def get_agent_types(
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get list of available agent types."""
    try:
        agent_types = await agent_coordinator.get_agent_types()
        return {"agent_types": agent_types}
    except Exception as e:
        logger.error(f"Error retrieving agent types: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent types: {str(e)}")

@router.get("/capabilities")
async def get_agent_capabilities(
    agent_coordinator: AgentCoordinator = Depends(get_agent_coordinator)
):
    """Get capabilities of the agent system."""
    try:
        capabilities = await agent_coordinator.get_capabilities()
        return capabilities
    except Exception as e:
        logger.error(f"Error retrieving agent capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent capabilities: {str(e)}") 