"""
Memory integration example.

This module demonstrates how to integrate the memory system with existing agent classes.
"""

import logging
import json
import sys
import os
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.types import Memory, MemoryType
from memory.manager import MemoryManager
from memory.agent_memory import MemoryCapableMixin
from memory.utils import memory_to_text

# Mock class to simulate the base Agent class
class Agent:
    """Mock Agent base class."""
    
    def __init__(self, agent_id, role):
        """Initialize the agent."""
        self.agent_id = agent_id
        self.role = role
        logger.info(f"Initialized base Agent with ID {agent_id} and role {role}")
    
    def send_message(self, receiver, content, message_type, correlation_id=None):
        """Mock sending a message."""
        correlation_id = correlation_id or str(uuid.uuid4())
        logger.info(f"Sending message to {receiver}: {message_type} (correlation_id: {correlation_id})")
        return correlation_id

# Original ProcessAnalystAgent class (simplified for illustration)
class OriginalProcessAnalystAgent(Agent):
    """
    Original ProcessAnalystAgent implementation without memory capabilities.
    This class is simplified for demonstration purposes.
    """
    
    def __init__(self, agent_id, agent_framework):
        """Initialize the process analyst agent."""
        super().__init__(agent_id, "PROCESS_ANALYST")
        self.agent_framework = agent_framework
        self.process_designs = {}  # process_id -> design
        logger.info(f"Initialized process analyst agent {agent_id}")
    
    def handle_design_request(self, task_id, message):
        """Handle a process design request."""
        logger.info(f"Handling design request for task {task_id}")
        
        # Extract requirements from message
        requirements = message.get("requirements", {})
        domain = requirements.get("domain", "general")
        process_name = requirements.get("process_name", "Unnamed Process")
        
        # Generate a process ID
        process_id = f"proc-{str(uuid.uuid4())[:8]}"
        
        # Create a simple process design
        design = {
            "process_id": process_id,
            "process_name": process_name,
            "domain": domain,
            "created_at": datetime.utcnow().isoformat(),
            "nodes": self._generate_nodes(requirements),
            "edges": self._generate_edges(requirements)
        }
        
        # Store the design
        self.process_designs[process_id] = design
        
        # Send back results
        self.send_message(
            receiver="coordinator",
            content={
                "task_id": task_id,
                "process_id": process_id,
                "status": "success"
            },
            message_type="design_results"
        )
        
        return process_id
    
    def get_process_design(self, process_id):
        """Get a process design from local storage."""
        return self.process_designs.get(process_id)
    
    def _generate_nodes(self, requirements):
        """Generate nodes for the process."""
        # Basic implementation without memory
        domain = requirements.get("domain", "general")
        
        # Generic nodes for any domain
        nodes = [
            {"id": "start", "type": "start", "label": "Start"},
            {"id": "end", "type": "end", "label": "End"}
        ]
        
        # If we have a banking domain, add some banking-specific nodes
        if domain == "banking":
            nodes.extend([
                {"id": "kyc", "type": "task", "label": "KYC Verification"},
                {"id": "account", "type": "task", "label": "Account Creation"}
            ])
        
        return nodes
    
    def _generate_edges(self, requirements):
        """Generate edges for the process."""
        # Basic implementation without memory
        domain = requirements.get("domain", "general")
        
        # Generic edges
        edges = []
        
        # If we have a banking domain, connect the banking-specific nodes
        if domain == "banking":
            edges = [
                {"source": "start", "target": "kyc", "type": "sequence"},
                {"source": "kyc", "target": "account", "type": "sequence"},
                {"source": "account", "target": "end", "type": "sequence"}
            ]
        else:
            # Default edge directly from start to end
            edges = [
                {"source": "start", "target": "end", "type": "sequence"}
            ]
        
        return edges

# Enhanced ProcessAnalystAgent with memory capabilities
class MemoryEnabledProcessAnalystAgent(Agent, MemoryCapableMixin):
    """
    Enhanced ProcessAnalystAgent with memory capabilities.
    """
    
    def __init__(self, agent_id, agent_framework):
        """Initialize the memory-enabled process analyst agent."""
        # Initialize the base agent
        Agent.__init__(self, agent_id, "PROCESS_ANALYST")
        
        # Initialize memory capabilities
        self._init_memory()
        
        self.agent_framework = agent_framework
        self.process_designs = {}  # process_id -> design
        logger.info(f"Initialized memory-enabled process analyst agent {agent_id}")
    
    def handle_design_request(self, task_id, message):
        """Handle a process design request with memory capabilities."""
        logger.info(f"Handling design request for task {task_id}")
        
        # Extract requirements from message
        requirements = message.get("requirements", {})
        domain = requirements.get("domain", "general")
        process_name = requirements.get("process_name", "Unnamed Process")
        
        # Generate a process ID
        process_id = f"proc-{str(uuid.uuid4())[:8]}"
        
        # Check memory for similar processes in this domain
        similar_processes = self._find_similar_processes(domain, requirements)
        
        # Create nodes and edges, possibly using similar processes as templates
        nodes = self._generate_nodes(requirements, similar_processes)
        edges = self._generate_edges(requirements, nodes, similar_processes)
        
        # Create a process design
        design = {
            "process_id": process_id,
            "process_name": process_name,
            "domain": domain,
            "created_at": datetime.utcnow().isoformat(),
            "nodes": nodes,
            "edges": edges,
            "requirements": requirements
        }
        
        # Store the design
        self.process_designs[process_id] = design
        
        # Remember this process design
        self._remember_design(process_id, process_name, design, domain, task_id)
        
        # Remember domain knowledge if we learned something new
        if domain and "specific_requirements" in requirements:
            self._store_domain_knowledge(domain, requirements)
        
        # Send back results
        self.send_message(
            receiver="coordinator",
            content={
                "task_id": task_id,
                "process_id": process_id,
                "status": "success",
                "used_memory": len(similar_processes) > 0
            },
            message_type="design_results"
        )
        
        return process_id
    
    def _find_similar_processes(self, domain, requirements):
        """Find similar processes from memory."""
        # Search for process memories with matching domain
        query = {
            "memory_type": MemoryType.PROCESS,
            "metadata": {"domain": domain}
        }
        
        process_memories = self._recall_memories(query, limit=5)
        logger.info(f"Found {len(process_memories)} similar processes in memory for domain '{domain}'")
        
        # Return the process designs from memory
        return [memory.content["design"] for memory in process_memories]
    
    def _remember_design(self, process_id, process_name, design, domain, task_id):
        """Remember a process design."""
        logger.info(f"Storing process design {process_id} in memory")
        
        memory_id = self._remember_process(
            process_id=process_id,
            process_name=process_name,
            content={
                "design": design,
                "task_id": task_id
            },
            importance=0.8
        )
        
        # Also add domain to metadata
        self._update_memory(
            memory_id=memory_id,
            metadata={"domain": domain}
        )
        
        return memory_id
    
    def _store_domain_knowledge(self, domain, requirements):
        """Remember domain-specific knowledge."""
        specific_requirements = requirements.get("specific_requirements", {})
        
        if not specific_requirements:
            return None
        
        logger.info(f"Storing domain knowledge for {domain} in memory")
        
        # Determine knowledge type based on requirements
        knowledge_type = "requirements"
        if "regulations" in specific_requirements:
            knowledge_type = "regulation"
        elif "best_practices" in specific_requirements:
            knowledge_type = "best_practice"
        
        memory_id = self._remember_domain_knowledge(
            domain=domain,
            knowledge_type=knowledge_type,
            content={
                "requirements": specific_requirements,
                "extracted_from": requirements
            },
            importance=0.9
        )
        
        return memory_id
    
    def _generate_nodes(self, requirements, similar_processes=None):
        """Generate nodes for the process, potentially using similar processes as templates."""
        domain = requirements.get("domain", "general")
        
        # Start with generic nodes
        nodes = [
            {"id": "start", "type": "start", "label": "Start"},
            {"id": "end", "type": "end", "label": "End"}
        ]
        
        # Check memory for domain knowledge about required nodes
        domain_knowledge = self._recall_domain_knowledge(domain)
        
        # If we have domain knowledge, use it to generate nodes
        if domain_knowledge:
            logger.info(f"Using domain knowledge to generate nodes for {domain}")
            nodes.extend(self._nodes_from_domain_knowledge(domain_knowledge, requirements))
        
        # If we have similar processes, use them as templates
        elif similar_processes:
            logger.info(f"Using similar processes as templates for {domain}")
            template_nodes = self._extract_template_nodes(similar_processes)
            nodes.extend(template_nodes)
        
        # Fallback to domain-specific nodes
        elif domain == "banking":
            nodes.extend([
                {"id": "kyc", "type": "task", "label": "KYC Verification"},
                {"id": "account", "type": "task", "label": "Account Creation"}
            ])
        
        return nodes
    
    def _nodes_from_domain_knowledge(self, domain_knowledge, requirements):
        """Generate nodes based on domain knowledge."""
        nodes = []
        
        for memory in domain_knowledge:
            content = memory.content
            
            # Extract required activities from domain knowledge
            if "requirements" in content:
                for req_key, req_value in content["requirements"].items():
                    if isinstance(req_value, list) and req_key in ["activities", "steps", "tasks"]:
                        for i, activity in enumerate(req_value):
                            node_id = f"task_{i+1}"
                            if isinstance(activity, str):
                                nodes.append({
                                    "id": node_id,
                                    "type": "task",
                                    "label": activity.replace("_", " ").title()
                                })
                            elif isinstance(activity, dict) and "name" in activity:
                                nodes.append({
                                    "id": node_id,
                                    "type": "task",
                                    "label": activity["name"].replace("_", " ").title()
                                })
        
        return nodes
    
    def _extract_template_nodes(self, similar_processes):
        """Extract template nodes from similar processes."""
        # Collect nodes from all similar processes
        all_nodes = []
        for process in similar_processes:
            if "nodes" in process:
                # Filter out start and end nodes
                process_nodes = [node for node in process["nodes"] 
                                if node.get("type") not in ["start", "end"]]
                all_nodes.extend(process_nodes)
        
        # Count node types and select the most common ones
        node_types = {}
        for node in all_nodes:
            node_type = node.get("label", "Unknown")
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
        
        # Sort by frequency
        sorted_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)
        
        # Create new nodes with the most common types
        template_nodes = []
        for i, (node_type, _) in enumerate(sorted_types[:5]):  # Limit to top 5
            template_nodes.append({
                "id": f"task_{i+1}",
                "type": "task",
                "label": node_type
            })
        
        return template_nodes
    
    def _generate_edges(self, requirements, nodes, similar_processes=None):
        """Generate edges for the process, potentially using similar processes as templates."""
        # Create a simple linear flow connecting all nodes
        edges = []
        
        # Sort nodes: start -> tasks -> end
        sorted_nodes = []
        start_node = None
        end_node = None
        task_nodes = []
        
        for node in nodes:
            if node["type"] == "start":
                start_node = node
            elif node["type"] == "end":
                end_node = node
            else:
                task_nodes.append(node)
        
        if start_node:
            sorted_nodes.append(start_node)
        sorted_nodes.extend(task_nodes)
        if end_node:
            sorted_nodes.append(end_node)
        
        # Connect nodes in sequence
        for i in range(len(sorted_nodes) - 1):
            source = sorted_nodes[i]["id"]
            target = sorted_nodes[i + 1]["id"]
            edges.append({
                "source": source,
                "target": target,
                "type": "sequence"
            })
        
        return edges
    
    def _recall_domain_knowledge(self, domain):
        """Recall domain knowledge from memory."""
        query = {
            "memory_type": MemoryType.DOMAIN_KNOWLEDGE,
            "metadata": {"domain": domain}
        }
        
        memories = self._recall_memories(query, limit=3)
        logger.info(f"Found {len(memories)} domain knowledge memories for {domain}")
        
        return memories
    
    def get_process_design(self, process_id):
        """Get a process design, from local storage or memory."""
        # First check local storage
        if process_id in self.process_designs:
            return self.process_designs[process_id]
        
        # If not found, check memory
        query = {
            "memory_type": MemoryType.PROCESS,
            "metadata": {"process_id": process_id}
        }
        
        memories = self._recall_memories(query, limit=1)
        
        if memories:
            design = memories[0].content.get("design")
            if design:
                # Cache it locally for future use
                self.process_designs[process_id] = design
                return design
        
        return None

def run_example():
    """Run the memory integration example."""
    # Create a mock agent framework for demonstration
    agent_framework = {}
    
    # Create original agent
    original_agent = OriginalProcessAnalystAgent(
        agent_id="original-analyst-001",
        agent_framework=agent_framework
    )
    
    # Create memory-enabled agent
    memory_agent = MemoryEnabledProcessAnalystAgent(
        agent_id="memory-analyst-001",
        agent_framework=agent_framework
    )
    
    # Test with a banking process design request
    banking_requirements = {
        "domain": "banking",
        "process_name": "Customer Onboarding",
        "specific_requirements": {
            "activities": [
                "identity_verification", 
                "document_collection",
                "risk_assessment",
                "account_creation",
                "welcome_email"
            ],
            "regulations": {
                "KYC": "Know Your Customer requirements",
                "AML": "Anti-Money Laundering checks"
            }
        }
    }
    
    # Process the request with both agents
    logger.info("\n=== Processing request with original agent ===\n")
    original_process_id = original_agent.handle_design_request(
        task_id="task-001",
        message={"requirements": banking_requirements}
    )
    
    logger.info("\n=== Processing request with memory-enabled agent ===\n")
    memory_process_id = memory_agent.handle_design_request(
        task_id="task-002",
        message={"requirements": banking_requirements}
    )
    
    # Simulate a second request to demonstrate memory usage
    logger.info("\n=== Processing second request with memory-enabled agent ===\n")
    second_process_id = memory_agent.handle_design_request(
        task_id="task-003",
        message={"requirements": {
            "domain": "banking",
            "process_name": "Corporate Account Opening",
            "specific_requirements": {
                "customer_type": "corporate"
            }
        }}
    )
    
    # Retrieve the designs
    original_design = original_agent.get_process_design(original_process_id)
    memory_design = memory_agent.get_process_design(memory_process_id)
    second_design = memory_agent.get_process_design(second_process_id)
    
    # Print the designs
    logger.info("\n=== Original agent design ===\n")
    logger.info(json.dumps(original_design, indent=2))
    
    logger.info("\n=== Memory-enabled agent first design ===\n")
    logger.info(json.dumps(memory_design, indent=2))
    
    logger.info("\n=== Memory-enabled agent second design ===\n")
    logger.info(json.dumps(second_design, indent=2))
    
    # Show memory context
    memory_context = memory_agent._get_memory_context()
    logger.info("\n=== Memory context ===\n")
    logger.info(f"Memory agent has {len(memory_context)} memories in context")
    logger.info(json.dumps(memory_context, indent=2))

if __name__ == "__main__":
    run_example() 