#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Agent Framework for managing multi-agent interactions.
This module implements the coordination mechanism for specialized AI agents.
"""

import os
import logging
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Enumeration of possible agent roles in the system."""
    COORDINATOR = "coordinator"
    PROCESS_ANALYST = "process_analyst"
    DOMAIN_EXPERT = "domain_expert"
    COMPLIANCE_OFFICER = "compliance_officer"
    OPTIMIZATION_SPECIALIST = "optimization_specialist"
    INTEGRATION_SPECIALIST = "integration_specialist"
    USER_PROXY = "user_proxy"

class AgentMessage:
    """Message passed between agents."""

    def __init__(self, 
                 sender: str,
                 receiver: str,
                 content: Any,
                 message_type: str = "request",
                 correlation_id: Optional[str] = None):
        """
        Initialize an agent message.
        
        Args:
            sender: ID of the sending agent
            receiver: ID of the receiving agent
            content: Content of the message
            message_type: Type of message (request, response, notification)
            correlation_id: Correlation ID for tracking conversation threads
        """
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = asyncio.get_event_loop().time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "message_type": self.message_type,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary representation."""
        message = cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            message_type=data["message_type"],
            correlation_id=data["correlation_id"]
        )
        message.timestamp = data["timestamp"]
        return message

class Agent:
    """Base agent class."""
    
    def __init__(self, 
                 agent_id: str,
                 role: AgentRole,
                 capabilities: List[str]):
        """
        Initialize an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role of the agent
            capabilities: List of capabilities the agent has
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.handlers = {}
        self.register_handler("default", self.default_message_handler)
    
    async def send_message(self, 
                          receiver: str, 
                          content: Any, 
                          message_type: str = "request",
                          correlation_id: Optional[str] = None) -> str:
        """
        Send a message to another agent.
        
        Args:
            receiver: ID of the receiving agent
            content: Content of the message
            message_type: Type of message
            correlation_id: Correlation ID for tracking conversation threads
            
        Returns:
            Correlation ID of the message
        """
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            message_type=message_type,
            correlation_id=correlation_id
        )
        
        # Use the class method to route the message
        await AgentFramework.route_message(message)
        
        return message.correlation_id
    
    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent.
        
        Args:
            message: The message to receive
        """
        await self.message_queue.put(message)
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Function to handle the message
        """
        self.handlers[message_type] = handler
    
    async def process_messages(self) -> None:
        """Process messages in the queue."""
        self.is_running = True
        
        while self.is_running:
            try:
                # Get a message from the queue
                message = await self.message_queue.get()
                
                # Find the appropriate handler
                handler = self.handlers.get(message.message_type, self.handlers["default"])
                
                # Handle the message
                await handler(message)
                
                # Mark task as done
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
    
    async def default_message_handler(self, message: AgentMessage) -> None:
        """Default handler for messages without a specific handler."""
        logger.warning(f"No handler for message type: {message.message_type}")
    
    async def stop(self) -> None:
        """Stop processing messages."""
        self.is_running = False

class ProcessAnalystAgent(Agent):
    """Agent specialized in analyzing business processes."""
    
    def __init__(self, agent_id: str):
        """
        Initialize a new process analyst agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PROCESS_ANALYST,
            capabilities=["process_analysis", "process_design", "bottleneck_detection"]
        )
        
        # Register message handlers
        self.register_handler("analyze_process", self.handle_analyze_process)
        self.register_handler("design_process", self.handle_design_process)
    
    async def handle_analyze_process(self, message: AgentMessage) -> None:
        """
        Handle a request to analyze a process.
        
        Args:
            message: The message containing the process data
        """
        process_data = message.content
        
        # Analyze the process using the GNN model
        analysis_results = self.gnn_model.analyze_process(process_data)
        
        # Send response back to sender
        await self.send_message(
            receiver=message.sender,
            content=analysis_results,
            message_type="analysis_results",
            correlation_id=message.correlation_id
        )
    
    async def handle_design_process(self, message: AgentMessage) -> None:
        """
        Handle requests to design a new process.
        
        Args:
            message: The message containing process requirements
        """
        process_data = message.content.get("process_data", {})
        task_id = message.content.get("task_id")
        workflow_id = message.content.get("workflow_id")
        
        logger.debug(f"Process analyst {self.agent_id} designing process")
        logger.debug(f"Process data: {process_data}")
        logger.debug(f"Task ID: {task_id}, Workflow ID: {workflow_id}")
        logger.debug(f"Message correlation_id: {message.correlation_id}")
        
        # Create a process design based on requirements
        process_design = {
            "metadata": {
                "name": process_data.get("process_name", "New Process"),
                "description": process_data.get("description", ""),
                "domain": process_data.get("domain", "general")
            },
            "graph": {
                "nodes": [
                    {
                        "id": "start",
                        "type": "start",
                        "name": "Start"
                    }
                ],
                "edges": []
            }
        }
        
        # Add activities from requirements
        activities = process_data.get("expected_activities", [])
        for i, activity in enumerate(activities):
            node_id = f"task_{i+1}"
            process_design["graph"]["nodes"].append({
                "id": node_id,
                "type": "task",
                "name": activity,
                "description": f"Perform {activity.lower()}"
            })
            
            # Connect to previous node
            prev_node = "start" if i == 0 else f"task_{i}"
            process_design["graph"]["edges"].append({
                "source": prev_node,
                "target": node_id
            })
        
        # Add end node and connect to last activity
        process_design["graph"]["nodes"].append({
            "id": "end",
            "type": "end",
            "name": "End"
        })
        if activities:
            process_design["graph"]["edges"].append({
                "source": f"task_{len(activities)}",
                "target": "end"
            })
        else:
            process_design["graph"]["edges"].append({
                "source": "start",
                "target": "end"
            })
        
        logger.debug(f"Created process design with {len(activities)} activities")
        logger.debug(f"Sending design results back to {message.sender} with correlation_id {message.correlation_id}")
        
        # Send response back with workflow information
        await self.send_message(
            receiver=message.sender,
            content={
                "process_design": process_design,
                "task_id": task_id,
                "workflow_id": workflow_id
            },
            message_type="design_results",
            correlation_id=message.correlation_id
        )

class CoordinatorAgent(Agent):
    """Agent responsible for coordinating workflows between agents."""
    
    def __init__(self, agent_id: str, agent_framework):
        """
        Initialize a new coordinator agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_framework: Reference to the agent framework
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            capabilities=["coordinate_workflow", "task_assignment", "result_aggregation"]
        )
        
        self.agent_framework = agent_framework
        self.active_workflows = {}
        
        # Register message handlers
        self.register_handler("start_workflow", self.handle_start_workflow)
        self.register_handler("analysis_results", self.handle_analysis_results)
        self.register_handler("bottleneck_results", self.handle_analysis_results)
        self.register_handler("efficiency_results", self.handle_analysis_results)
        self.register_handler("design_results", self.handle_design_results)
    
    async def handle_start_workflow(self, message: AgentMessage) -> None:
        """
        Handle a request to start a new workflow.
        
        Args:
            message: The message containing workflow details
        """
        workflow_data = message.content
        workflow_type = workflow_data.get("workflow_type")
        
        # Log the incoming workflow request
        logger.debug(f"Coordinator agent {self.agent_id} received start_workflow request: {workflow_type}")
        logger.debug(f"Workflow data: {workflow_data}")
        logger.debug(f"Message correlation_id: {message.correlation_id}")
        
        # Generate a unique ID for this workflow if not provided
        workflow_id = message.correlation_id or str(uuid.uuid4())
        
        # Create workflow state
        self.active_workflows[workflow_id] = {
            "workflow_type": workflow_type,
            "requester": message.sender,
            "status": "in_progress",
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "correlation_id": message.correlation_id,
            "pending_tasks": [],
            "completed_tasks": [],
            "results": {}
        }
        
        logger.debug(f"Created new workflow: {workflow_id} of type {workflow_type}")
        
        process_data = workflow_data.get("process_data", {})
        
        # Initialize success flag
        success = False
        
        # Handle different workflow types
        if workflow_type == "process_analysis":
            # Get process analyst agent for analysis
            analyst_agent = self.agent_framework.get_agent_by_role(AgentRole.PROCESS_ANALYST)
            
            if analyst_agent:
                task_id = str(uuid.uuid4())
                self.active_workflows[workflow_id]["pending_tasks"].append(task_id)
                
                # Send analysis request
                await self.send_message(
                    receiver=analyst_agent.agent_id,
                    content={
                        "process_data": process_data,
                        "task_id": task_id,
                        "workflow_id": workflow_id
                    },
                    message_type="analyze_process",
                    correlation_id=task_id
                )
                success = True
            else:
                logger.error(f"No process analyst agent available for workflow {workflow_id}")
                await self.send_message(
                    receiver=message.sender,
                    content={
                        "status": "error",
                        "error": "No process analyst agent available",
                        "workflow_id": workflow_id
                    },
                    message_type="workflow_status",
                    correlation_id=message.correlation_id
                )
        
        elif workflow_type == "process_design":
            # Get process analyst agent for design
            analyst_agent = self.agent_framework.get_agent_by_role(AgentRole.PROCESS_ANALYST)
            
            if analyst_agent:
                task_id = str(uuid.uuid4())
                self.active_workflows[workflow_id]["pending_tasks"].append(task_id)
                
                logger.debug(f"Creating design process task {task_id} for workflow {workflow_id}")
                
                # Send process design request
                await self.send_message(
                    receiver=analyst_agent.agent_id,
                    content={
                        "process_data": process_data,
                        "task_id": task_id,
                        "workflow_id": workflow_id
                    },
                    message_type="design_process",
                    correlation_id=task_id
                )
                success = True
            else:
                logger.error(f"No process analyst agent available for workflow {workflow_id}")
                await self.send_message(
                    receiver=message.sender,
                    content={
                        "status": "error",
                        "error": "No process analyst agent available",
                        "workflow_id": workflow_id
                    },
                    message_type="workflow_status",
                    correlation_id=message.correlation_id
                )
        else:
            # Unsupported workflow type
            logger.warning(f"Unsupported workflow type: {workflow_type}")
            await self.send_message(
                receiver=message.sender,
                content={"error": f"Unsupported workflow type: {workflow_type}"},
                message_type="workflow_error",
                correlation_id=message.correlation_id
            )
        
        # If workflow failed to start, update status
        if not success:
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["end_time"] = datetime.utcnow().isoformat()
        
        # Send acknowledgment to requester
        await self.send_message(
            receiver=message.sender,
            content={
                "status": "started" if success else "failed",
                "workflow_id": workflow_id
            },
            message_type="workflow_status",
            correlation_id=message.correlation_id
        )
    
    async def handle_analysis_results(self, message: AgentMessage) -> None:
        """
        Handle analysis results from a process analyst.
        
        Args:
            message: The message containing the analysis results
        """
        correlation_id = message.correlation_id
        
        if correlation_id in self.active_workflows:
            workflow = self.active_workflows[correlation_id]
            
            # Update workflow with results
            workflow["results"]["analysis"] = message.content
            
            # Update step status
            workflow["steps"][workflow["current_step"]]["status"] = "completed"
            workflow["current_step"] += 1
            
            # Check if workflow is complete
            if self._is_workflow_complete(workflow):
                workflow["status"] = "completed"
                
                # Send results to requester
                await self.send_message(
                    receiver=workflow["requester"],
                    content={"workflow_id": correlation_id, "results": workflow["results"]},
                    message_type="workflow_completed",
                    correlation_id=correlation_id
                )
    
    def _is_workflow_complete(self, workflow: Dict[str, Any]) -> bool:
        """
        Check if a workflow is complete.
        
        Args:
            workflow: The workflow to check
            
        Returns:
            True if the workflow is complete, False otherwise
        """
        return workflow["current_step"] >= len(workflow["steps"])
    
    async def handle_design_results(self, message: AgentMessage) -> None:
        """
        Handle results from process design.
        
        Args:
            message: The message containing the process design results
        """
        task_id = message.correlation_id
        
        logger.debug(f"Coordinator agent {self.agent_id} received design results")
        logger.debug(f"Message content: {message.content}")
        logger.debug(f"Task ID (correlation_id): {task_id}")
        
        # Find the workflow this task belongs to
        workflow_id = None
        for wf_id, workflow in self.active_workflows.items():
            if task_id in workflow["pending_tasks"]:
                workflow_id = wf_id
                break
        
        if not workflow_id:
            logger.warning(f"Received results for unknown task {task_id}")
            logger.debug(f"Active workflows: {self.active_workflows.keys()}")
            return
        
        logger.debug(f"Found workflow {workflow_id} for task {task_id}")
        
        # Update workflow state
        workflow = self.active_workflows[workflow_id]
        workflow["pending_tasks"].remove(task_id)
        workflow["completed_tasks"].append(task_id)
        
        # Store the process design in the results
        process_design = message.content.get("process_design")
        workflow["results"]["process_design"] = process_design
        
        # For process design workflows, we can mark as completed after receiving design results
        workflow["status"] = "completed"
        workflow["end_time"] = datetime.utcnow().isoformat()
        
        logger.debug(f"Workflow {workflow_id} is complete")
        
        # Create a Process object to store in the vector store
        try:
            # Generate a process ID
            process_id = str(uuid.uuid4())
            
            # Create a process from the design
            metadata = process_design.get("metadata", {})
            graph = process_design.get("graph", {})
            
            # Store the process in the vector store
            process_data = {
                "process_id": process_id,
                "name": metadata.get("name", "Unnamed Process"),
                "description": metadata.get("description", ""),
                "domain": metadata.get("domain", "general"),
                "complexity": 0.5,  # Default complexity
                "upload_time": datetime.utcnow().isoformat(),
                "graph": graph
            }
            
            # In a real implementation, we would:
            # 1. Store the process in the vector store
            # self.agent_framework.vector_store.add_process(process_id, process_data)
            
            # 2. Generate embeddings using the GNN model
            # embeddings = self.agent_framework.gnn_model.encode_process(process_data)
            # self.agent_framework.vector_store.update_embeddings(process_id, embeddings)
            
            # Add to the results
            workflow["results"]["process_id"] = process_id
            
            logger.info(f"Created new process {process_id}: {metadata.get('name')}")
        except Exception as e:
            logger.error(f"Error creating process: {str(e)}")
        
        # Send completion message to requester
        await self.send_message(
            receiver=workflow["requester"],
            content={
                "status": "completed",
                "workflow_id": workflow_id,
                "results": workflow["results"]
            },
            message_type="workflow_completed",
            correlation_id=workflow_id
        )

class AgentFramework:
    """Framework for managing and coordinating agents."""
    
    # Class variable for message routing
    _agent_registry = {}
    
    def __init__(self, gnn_model, vector_store):
        """
        Initialize the agent framework.
        
        Args:
            gnn_model: GNN model for process analysis
            vector_store: Vector store for process data
        """
        self.gnn_model = gnn_model
        self.vector_store = vector_store
        self.agents = {}
        self.tasks = []
        
        # Set up agents
        self._setup_agents()
    
    def _setup_agents(self):
        """Set up the initial set of agents."""
        # Create process analyst agent
        process_analyst = ProcessAnalystAgent(
            agent_id="process_analyst_1"
        )
        self.register_agent(process_analyst)
        
        # Create coordinator agent
        coordinator = CoordinatorAgent(
            agent_id="coordinator_1",
            agent_framework=self
        )
        self.register_agent(coordinator)
    
    def register_agent(self, agent: Agent):
        """
        Register an agent with the framework.
        
        Args:
            agent: The agent to register
        """
        self.agents[agent.agent_id] = agent
        AgentFramework._agent_registry[agent.agent_id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.
        
        Args:
            agent_id: ID of the agent to get
            
        Returns:
            The agent with the given ID, or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_agent_by_role(self, role: AgentRole) -> Optional[Agent]:
        """
        Get the first agent with a given role.
        
        Args:
            role: Role to search for
            
        Returns:
            The first agent with the given role, or None if not found
        """
        for agent in self.agents.values():
            if agent.role == role:
                return agent
        return None
    
    def get_agents_by_role(self, role: AgentRole) -> List[Agent]:
        """
        Get all agents with a given role.
        
        Args:
            role: Role to search for
            
        Returns:
            List of agents with the given role
        """
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def get_agents_with_capability(self, capability: str) -> List[Agent]:
        """
        Get all agents with a given capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agents with the given capability
        """
        return [
            agent for agent in self.agents.values()
            if capability in agent.capabilities
        ]
    
    @classmethod
    async def route_message(cls, message: AgentMessage) -> None:
        """
        Route a message to its intended recipient.
        
        Args:
            message: The message to route
        """
        receiver_id = message.receiver
        
        if receiver_id in cls._agent_registry:
            receiver = cls._agent_registry[receiver_id]
            await receiver.receive_message(message)
        else:
            logger.warning(f"No agent found with ID {receiver_id} to receive message")
    
    async def start_agents(self):
        """Start all agents."""
        for agent in self.agents.values():
            task = asyncio.create_task(agent.process_messages())
            self.tasks.append(task)
    
    async def stop_agents(self):
        """Stop all agents."""
        for agent in self.agents.values():
            await agent.stop()
        
        for task in self.tasks:
            task.cancel()
    
    async def process_request(self, request_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request using the agent system.
        
        Args:
            request_type: Type of request
            data: Request data
            
        Returns:
            Response data
        """
        # Create a future to wait for the response
        response_future = asyncio.Future()
        
        # Get coordinator agent
        coordinator = self.get_agent_by_role(AgentRole.COORDINATOR)
        
        if not coordinator:
            raise ValueError("No coordinator agent available")
        
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Register response handler
        async def response_handler(message: AgentMessage):
            if not response_future.done():
                response_future.set_result(message.content)
        
        # Create a temporary agent to receive the response
        temp_agent = Agent(
            agent_id=f"temp_{correlation_id}",
            role=AgentRole.USER_PROXY,
            capabilities=[]
        )
        temp_agent.register_handler("workflow_completed", response_handler)
        temp_agent.register_handler("workflow_error", response_handler)
        self.register_agent(temp_agent)
        
        # Start the temporary agent
        temp_task = asyncio.create_task(temp_agent.process_messages())
        
        try:
            # Send request to coordinator
            await coordinator.send_message(
                receiver=coordinator.agent_id,
                content={"workflow_type": request_type, **data},
                message_type="start_workflow",
                correlation_id=correlation_id
            )
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        finally:
            # Clean up
            temp_task.cancel()
            self.agents.pop(temp_agent.agent_id, None)
            AgentFramework._agent_registry.pop(temp_agent.agent_id, None) 