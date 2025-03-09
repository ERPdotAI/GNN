"""
Memory system test script.

This script demonstrates the functionality of the memory system.
"""

import logging
import json
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Any

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
from memory.storage import InMemoryStorage, FileStorage
from memory.manager import MemoryManager
from memory.utils import memory_to_text, memories_to_text, deduplicate_memories
from memory.agent_memory import MemoryCapableMixin

class TestAgent(MemoryCapableMixin):
    """Test agent that demonstrates memory capabilities."""
    
    def __init__(self, agent_id: str):
        """Initialize the test agent."""
        self.agent_id = agent_id
        self._init_memory()
        logger.info(f"Initialized test agent with ID {agent_id}")
    
    def create_test_memories(self):
        """Create some test memories."""
        logger.info("Creating test memories...")
        
        # Create a process memory
        process_id = "proc-12345"
        process_memory_id = self._remember_process(
            process_id=process_id,
            process_name="Customer Onboarding",
            content={
                "status": "active",
                "steps": ["identity_verification", "account_creation", "welcome_email"],
                "completion": 0.33
            },
            importance=0.8
        )
        logger.info(f"Created process memory: {process_memory_id}")
        
        # Create a task memory
        task_id = "task-67890"
        task_memory_id = self._remember_task(
            task_id=task_id,
            task_name="Identity Verification",
            content={
                "status": "pending",
                "user_input_required": True,
                "documents": ["passport", "utility_bill"]
            },
            importance=0.7
        )
        logger.info(f"Created task memory: {task_memory_id}")
        
        # Create a user preference memory
        user_preference_id = self._remember_user_preference(
            user_id="user-54321",
            preference_type="workflow_style",
            content={
                "preferred_layout": "sequential",
                "show_progress": True,
                "notification_frequency": "daily"
            },
            importance=0.6
        )
        logger.info(f"Created user preference memory: {user_preference_id}")
        
        # Create an interaction memory
        interaction_id = self._remember_interaction(
            user_id="user-54321",
            interaction_type="feedback",
            content={
                "rating": 4,
                "comment": "Process was smooth but took longer than expected",
                "improvement_areas": ["speed", "communication"]
            },
            importance=0.5
        )
        logger.info(f"Created interaction memory: {interaction_id}")
        
        # Create domain knowledge memory
        domain_knowledge_id = self._remember_domain_knowledge(
            domain="banking",
            knowledge_type="regulation",
            content={
                "name": "KYC Compliance",
                "requirements": ["identity_verification", "address_verification", "source_of_funds"],
                "update_date": "2023-01-15"
            },
            importance=0.9
        )
        logger.info(f"Created domain knowledge memory: {domain_knowledge_id}")
        
        return {
            "process_id": process_id,
            "process_memory_id": process_memory_id,
            "task_id": task_id,
            "task_memory_id": task_memory_id,
            "user_preference_id": user_preference_id,
            "interaction_id": interaction_id,
            "domain_knowledge_id": domain_knowledge_id
        }
    
    def retrieve_memories(self, memory_ids):
        """Retrieve memories by ID."""
        logger.info("Retrieving memories by ID...")
        
        for name, memory_id in memory_ids.items():
            if name.endswith("_id") and not name.startswith("process_id") and not name.startswith("task_id"):
                memory = self._recall(memory_id)
                if memory:
                    logger.info(f"Retrieved {name}:")
                    logger.info(memory_to_text(memory))
                else:
                    logger.warning(f"Failed to retrieve {name}")
    
    def search_memories(self, process_id, task_id):
        """Search for memories by different criteria."""
        logger.info("Searching for memories...")
        
        # Get process memories
        process_memories = self._recall_process_related(process_id)
        logger.info(f"Process memories ({len(process_memories)}):")
        logger.info(memories_to_text(process_memories))
        
        # Get task memories
        task_memories = self._recall_task_related(task_id)
        logger.info(f"Task memories ({len(task_memories)}):")
        logger.info(memories_to_text(task_memories))
        
        # Get all memories by type
        domain_memories = self._recall_by_type(MemoryType.DOMAIN_KNOWLEDGE)
        logger.info(f"Domain knowledge memories ({len(domain_memories)}):")
        logger.info(memories_to_text(domain_memories))
        
        # Get memory context
        context = self._get_memory_context()
        logger.info(f"Memory context ({len(context)} items):")
        logger.info(json.dumps(context, indent=2))
    
    def update_memory_demo(self, memory_id):
        """Demonstrate memory updates."""
        logger.info("Updating memory...")
        
        # Retrieve the original memory
        original = self._recall(memory_id)
        if not original:
            logger.warning(f"Failed to retrieve memory {memory_id}")
            return
        
        logger.info("Original memory:")
        logger.info(memory_to_text(original))
        
        # Update the memory
        update_result = self._update_memory(
            memory_id=memory_id,
            content={**original.content, "status": "completed", "completion": 1.0},
            importance=0.9
        )
        
        if update_result:
            logger.info("Memory updated successfully")
            
            # Retrieve the updated memory
            updated = self._recall(memory_id)
            logger.info("Updated memory:")
            logger.info(memory_to_text(updated))
        else:
            logger.warning("Failed to update memory")
    
    def delete_memory_demo(self, memory_id):
        """Demonstrate memory deletion."""
        logger.info("Deleting memory...")
        
        delete_result = self._forget(memory_id)
        
        if delete_result:
            logger.info(f"Memory {memory_id} deleted successfully")
            
            # Try to retrieve the deleted memory
            deleted = self._recall(memory_id)
            if deleted:
                logger.warning(f"Memory {memory_id} still exists after deletion!")
            else:
                logger.info(f"Memory {memory_id} no longer exists, as expected")
        else:
            logger.warning(f"Failed to delete memory {memory_id}")

def run_file_storage_test():
    """Test file-based storage."""
    logger.info("\n\n=== Testing file-based storage ===\n")
    
    # Create a file storage manager
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_storage")
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    
    file_storage = FileStorage(storage_dir=storage_dir)
    manager = MemoryManager(storage=file_storage)
    
    # Create a test memory
    memory_id = manager.create_memory(
        content={"test": "file_storage", "timestamp": datetime.utcnow().isoformat()},
        memory_type=MemoryType.PROCESS,
        metadata={"source": "test_script"}
    )
    
    logger.info(f"Created test memory in file storage: {memory_id}")
    
    # Retrieve the memory
    memory = manager.get_memory(memory_id)
    if memory:
        logger.info("Retrieved memory from file storage:")
        logger.info(memory_to_text(memory))
    else:
        logger.warning("Failed to retrieve memory from file storage")
    
    # Verify the files were created
    memory_path = os.path.join(storage_dir, "memories", f"{memory_id}.json")
    index_path = os.path.join(storage_dir, "index.json")
    
    logger.info(f"Memory file exists: {os.path.exists(memory_path)}")
    logger.info(f"Index file exists: {os.path.exists(index_path)}")
    
    # Load the index file
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            index = json.load(f)
            logger.info(f"Index contains {len(index)} memories")
    
    logger.info("File storage test complete")

def main():
    """Run the memory system test."""
    logger.info("Starting memory system test...")
    
    # Create a test agent
    agent = TestAgent(agent_id="test-agent-001")
    
    # Create test memories
    memory_ids = agent.create_test_memories()
    
    # Retrieve memories
    agent.retrieve_memories(memory_ids)
    
    # Search for memories
    agent.search_memories(memory_ids["process_id"], memory_ids["task_id"])
    
    # Update a memory
    agent.update_memory_demo(memory_ids["process_memory_id"])
    
    # Delete a memory
    agent.delete_memory_demo(memory_ids["interaction_id"])
    
    # Test file storage
    run_file_storage_test()
    
    logger.info("Memory system test completed successfully")

if __name__ == "__main__":
    main() 