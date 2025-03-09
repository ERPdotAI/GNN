"""
Tests for the memory system components in AgenticProcessGNN.

This module contains tests for memory types, storage, manager and integration with agents.
"""

import os
import uuid
import pytest
import shutil
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.memory.types import Memory, MemoryType
from src.memory.storage import InMemoryStorage, FileStorage
from src.memory.manager import MemoryManager
from src.memory.agent_memory import MemoryCapableMixin
from src.memory.utils import format_memory_timestamp, parse_memory_timestamp


class TestMemoryTypes:
    """Tests for the Memory class and MemoryType enum."""
    
    def test_memory_creation(self):
        """Test that a Memory object can be created with the expected attributes."""
        memory = Memory(
            content="This is a test memory",
            memory_type=MemoryType.PROCESS,
            metadata={"process_id": "test-123"}
        )
        
        assert memory.content == "This is a test memory"
        assert memory.memory_type == MemoryType.PROCESS
        assert memory.metadata == {"process_id": "test-123"}
        assert isinstance(memory.memory_id, str)
        assert isinstance(memory.timestamp, datetime)
        assert memory.importance == 0.5  # default value
        
    def test_memory_to_dict(self):
        """Test converting a Memory object to a dictionary."""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        memory = Memory(
            content="Test process",
            memory_type=MemoryType.PROCESS,
            memory_id=memory_id,
            timestamp=timestamp,
            metadata={"domain": "banking"},
            importance=0.8
        )
        
        memory_dict = memory.to_dict()
        
        assert memory_dict["content"] == "Test process"
        assert memory_dict["memory_type"] == "PROCESS"
        assert memory_dict["memory_id"] == memory_id
        assert memory_dict["metadata"] == {"domain": "banking"}
        assert memory_dict["importance"] == 0.8
        assert isinstance(memory_dict["timestamp"], str)
        
    def test_memory_from_dict(self):
        """Test creating a Memory object from a dictionary."""
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        timestamp_str = format_memory_timestamp(timestamp)
        
        memory_dict = {
            "content": "Test domain knowledge",
            "memory_type": "DOMAIN_KNOWLEDGE",
            "memory_id": memory_id,
            "timestamp": timestamp_str,
            "metadata": {"domain": "healthcare"},
            "importance": 0.9
        }
        
        memory = Memory.from_dict(memory_dict)
        
        assert memory.content == "Test domain knowledge"
        assert memory.memory_type == MemoryType.DOMAIN_KNOWLEDGE
        assert memory.memory_id == memory_id
        assert memory.metadata == {"domain": "healthcare"}
        assert memory.importance == 0.9
        # Check if timestamps are approximately equal (within 1 second)
        assert abs((memory.timestamp - timestamp).total_seconds()) < 1


class TestMemoryStorage:
    """Tests for memory storage implementations."""
    
    def test_in_memory_storage(self):
        """Test InMemoryStorage operations."""
        storage = InMemoryStorage()
        
        # Create test memory
        memory = Memory(
            content="Storage test memory",
            memory_type=MemoryType.TASK,
            metadata={"task_id": "task-001"}
        )
        
        # Test add
        storage.add(memory)
        assert len(storage.memories) == 1
        
        # Test get
        retrieved = storage.get(memory.memory_id)
        assert retrieved.content == "Storage test memory"
        assert retrieved.memory_type == MemoryType.TASK
        
        # Test update
        memory.content = "Updated content"
        storage.update(memory)
        updated = storage.get(memory.memory_id)
        assert updated.content == "Updated content"
        
        # Test delete
        storage.delete(memory.memory_id)
        assert len(storage.memories) == 0
        
    def test_file_storage(self):
        """Test FileStorage operations."""
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        try:
            storage = FileStorage(directory_path=temp_dir)
            
            # Create test memory
            memory = Memory(
                content="File storage test",
                memory_type=MemoryType.USER_PREFERENCE,
                metadata={"user_id": "user-001"}
            )
            
            # Test add
            storage.add(memory)
            
            # Test get
            retrieved = storage.get(memory.memory_id)
            assert retrieved.content == "File storage test"
            assert retrieved.memory_type == MemoryType.USER_PREFERENCE
            
            # Test update
            memory.content = "Updated file content"
            storage.update(memory)
            updated = storage.get(memory.memory_id)
            assert updated.content == "Updated file content"
            
            # Test delete
            storage.delete(memory.memory_id)
            with pytest.raises(KeyError):
                storage.get(memory.memory_id)
                
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)


class TestMemoryManager:
    """Tests for the MemoryManager class."""
    
    def test_memory_manager_with_in_memory_storage(self):
        """Test MemoryManager with InMemoryStorage."""
        storage = InMemoryStorage()
        manager = MemoryManager(storage=storage)
        
        # Test add_memory
        memory_id = manager.add_memory(
            content="Manager test memory",
            memory_type=MemoryType.PROCESS,
            metadata={"process_name": "Test Process"}
        )
        
        # Test get_memory
        memory = manager.get_memory(memory_id)
        assert memory.content == "Manager test memory"
        
        # Test update_memory
        manager.update_memory(
            memory_id=memory_id,
            content="Updated manager test memory"
        )
        updated = manager.get_memory(memory_id)
        assert updated.content == "Updated manager test memory"
        
        # Test delete_memory
        manager.delete_memory(memory_id)
        with pytest.raises(KeyError):
            manager.get_memory(memory_id)
            
    def test_memory_manager_query(self):
        """Test querying memories from MemoryManager."""
        storage = InMemoryStorage()
        manager = MemoryManager(storage=storage)
        
        # Add test memories
        manager.add_memory(
            content="Banking process 1",
            memory_type=MemoryType.PROCESS,
            metadata={"domain": "banking", "process_id": "p1"}
        )
        manager.add_memory(
            content="Banking process 2",
            memory_type=MemoryType.PROCESS,
            metadata={"domain": "banking", "process_id": "p2"}
        )
        manager.add_memory(
            content="Healthcare process",
            memory_type=MemoryType.PROCESS,
            metadata={"domain": "healthcare", "process_id": "p3"}
        )
        
        # Test query by type
        process_memories = manager.query_memories(memory_type=MemoryType.PROCESS)
        assert len(process_memories) == 3
        
        # Test query by metadata
        banking_memories = manager.query_memories(
            memory_type=MemoryType.PROCESS,
            metadata_filter={"domain": "banking"}
        )
        assert len(banking_memories) == 2
        
        # Test query by content (exact match)
        healthcare_memories = manager.query_memories(
            content="Healthcare process",
            exact_match=True
        )
        assert len(healthcare_memories) == 1
        
        # Test query by content (substring)
        banking_memories = manager.query_memories(
            content="Banking",
            exact_match=False
        )
        assert len(banking_memories) == 2


class MockAgent:
    """Mock agent class for testing the MemoryCapableMixin."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        

class TestMemoryCapableMixin:
    """Tests for the MemoryCapableMixin."""
    
    def test_memory_capable_mixin(self):
        """Test integrating memory capabilities into an agent."""
        # Create a test agent class with memory capabilities
        class TestAgent(MockAgent, MemoryCapableMixin):
            def __init__(self, agent_id):
                MockAgent.__init__(self, agent_id)
                MemoryCapableMixin.__init__(self)
                
        # Create test agent
        agent = TestAgent(agent_id="test-agent-001")
        
        # Test remember method
        memory_id = agent.remember(
            content="Agent memory test",
            memory_type=MemoryType.INTERACTION,
            metadata={"user_id": "user-001"}
        )
        
        # Test recall method
        memories = agent.recall(memory_type=MemoryType.INTERACTION)
        assert len(memories) == 1
        assert memories[0].content == "Agent memory test"
        
        # Test forget method
        agent.forget(memory_id)
        post_forget_memories = agent.recall(memory_type=MemoryType.INTERACTION)
        assert len(post_forget_memories) == 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 