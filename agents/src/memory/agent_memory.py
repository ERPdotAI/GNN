"""
Agent memory integration.

This module provides a mixin class that adds memory capabilities to agents.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import uuid

from .manager import MemoryManager
from .types import Memory, MemoryType

# Setup logging
logger = logging.getLogger(__name__)

class MemoryCapableMixin:
    """
    Mixin class that adds memory capabilities to agents.
    
    To use this mixin:
    1. Add it as a base class to your agent class
    2. Call self._init_memory() in your agent's __init__ method
    3. Use memory methods as needed in your agent's implementation
    """
    
    def _init_memory(self, memory_manager: Optional[MemoryManager] = None):
        """
        Initialize memory capabilities.
        
        Args:
            memory_manager: Optional memory manager to use (if None, a new one will be created)
        """
        # Store the memory manager
        self._memory_manager = memory_manager or MemoryManager()
        
        # Keep track of retrieved memories for context
        self._recent_memories = []
        self._memory_context_size = 5  # Number of recent memories to keep in context
        
        logger.debug(f"Initialized memory capabilities for agent {self.agent_id}")
    
    def _remember(self, content: Dict[str, Any], memory_type: MemoryType, 
                metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> str:
        """
        Create a new memory.
        
        Args:
            content: The content of the memory
            memory_type: Type of memory
            metadata: Additional metadata for the memory (optional)
            importance: Importance score (0.0-1.0)
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        memory_id = self._memory_manager.create_memory(
            content=content,
            memory_type=memory_type,
            agent_id=self.agent_id,
            metadata=metadata or {},
            importance=importance
        )
        
        return memory_id
    
    def _recall(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory, or None if not found
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        memory = self._memory_manager.get_memory(memory_id)
        
        if memory:
            self._add_to_recent_memories(memory)
        
        return memory
    
    def _recall_memories(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching a query.
        
        Args:
            query: Dictionary of search criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        # Always filter by this agent's ID
        query["agent_id"] = self.agent_id
        
        memories = self._memory_manager.search_memories(query, limit)
        
        # Add the first memory to recent memories for context
        if memories:
            self._add_to_recent_memories(memories[0])
        
        return memories
    
    def _recall_by_type(self, memory_type: MemoryType, limit: int = 10) -> List[Memory]:
        """
        Retrieve memories of a specific type.
        
        Args:
            memory_type: Type of memory to retrieve
            limit: Maximum number of results to return
            
        Returns:
            List of memories
        """
        return self._recall_memories({"memory_type": memory_type}, limit)
    
    def _recall_process_related(self, process_id: str, limit: int = 10) -> List[Memory]:
        """
        Retrieve memories related to a specific process.
        
        Args:
            process_id: ID of the process
            limit: Maximum number of results to return
            
        Returns:
            List of memories
        """
        query = {
            "memory_type": MemoryType.PROCESS,
            "metadata": {"process_id": process_id}
        }
        
        return self._recall_memories(query, limit)
    
    def _recall_task_related(self, task_id: str, limit: int = 10) -> List[Memory]:
        """
        Retrieve memories related to a specific task.
        
        Args:
            task_id: ID of the task
            limit: Maximum number of results to return
            
        Returns:
            List of memories
        """
        query = {
            "memory_type": MemoryType.TASK,
            "metadata": {"task_id": task_id}
        }
        
        return self._recall_memories(query, limit)
    
    def _remember_process(self, process_id: str, process_name: str, 
                        content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory related to a process.
        
        Args:
            process_id: ID of the process
            process_name: Name of the process
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        return self._memory_manager.create_process_memory(
            agent_id=self.agent_id,
            process_id=process_id,
            process_name=process_name,
            content=content,
            importance=importance
        )
    
    def _remember_task(self, task_id: str, task_name: str,
                     content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory related to a task.
        
        Args:
            task_id: ID of the task
            task_name: Name of the task
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        return self._memory_manager.create_task_memory(
            agent_id=self.agent_id,
            task_id=task_id,
            task_name=task_name,
            content=content,
            importance=importance
        )
    
    def _remember_interaction(self, user_id: str, interaction_type: str,
                          content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory about an interaction.
        
        Args:
            user_id: ID of the user
            interaction_type: Type of interaction (e.g., "feedback", "request")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        return self._memory_manager.create_interaction_memory(
            agent_id=self.agent_id,
            user_id=user_id,
            interaction_type=interaction_type,
            content=content,
            importance=importance
        )
    
    def _remember_user_preference(self, user_id: str, preference_type: str,
                              content: Dict[str, Any], importance: float = 0.7) -> str:
        """
        Create a memory about a user preference.
        
        Args:
            user_id: ID of the user
            preference_type: Type of preference (e.g., "workflow_style", "ui_preference")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        return self._memory_manager.create_user_preference_memory(
            agent_id=self.agent_id,
            user_id=user_id,
            preference_type=preference_type,
            content=content,
            importance=importance
        )
    
    def _remember_domain_knowledge(self, domain: str, knowledge_type: str,
                              content: Dict[str, Any], importance: float = 0.8) -> str:
        """
        Create a memory about domain knowledge.
        
        Args:
            domain: Domain name (e.g., "banking", "healthcare")
            knowledge_type: Type of knowledge (e.g., "best_practice", "regulation")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        return self._memory_manager.create_domain_knowledge_memory(
            agent_id=self.agent_id,
            domain=domain,
            knowledge_type=knowledge_type,
            content=content,
            importance=importance
        )
    
    def _forget(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        result = self._memory_manager.delete_memory(memory_id)
        
        # Remove from recent memories if present
        self._recent_memories = [m for m in self._recent_memories if m.memory_id != memory_id]
        
        return result
    
    def _update_memory(self, memory_id: str, content: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None, 
                    importance: Optional[float] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (optional)
            metadata: New metadata (optional)
            importance: New importance (optional)
            
        Returns:
            True if updated, False if not found
        """
        if not hasattr(self, '_memory_manager'):
            raise RuntimeError("Memory capabilities not initialized. Call _init_memory() first.")
        
        result = self._memory_manager.update_memory(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            importance=importance
        )
        
        # Update in recent memories if present
        if result:
            updated_memory = self._memory_manager.get_memory(memory_id)
            if updated_memory:
                for i, memory in enumerate(self._recent_memories):
                    if memory.memory_id == memory_id:
                        self._recent_memories[i] = updated_memory
                        break
        
        return result
    
    def _add_to_recent_memories(self, memory: Memory) -> None:
        """
        Add a memory to the recent memories list.
        
        Args:
            memory: Memory to add
        """
        # Check if this memory is already in the recent list
        for i, m in enumerate(self._recent_memories):
            if m.memory_id == memory.memory_id:
                # Move it to the front (most recent)
                self._recent_memories.pop(i)
                self._recent_memories.insert(0, memory)
                return
        
        # Add to front of list
        self._recent_memories.insert(0, memory)
        
        # Trim list if needed
        if len(self._recent_memories) > self._memory_context_size:
            self._recent_memories = self._recent_memories[:self._memory_context_size]
    
    def _get_memory_context(self) -> List[Dict[str, Any]]:
        """
        Get the current memory context (recent memories).
        
        Returns:
            List of recent memories as dictionaries
        """
        return [
            {
                "memory_id": m.memory_id,
                "memory_type": m.memory_type.value,
                "content": m.content,
                "metadata": m.metadata,
                "importance": m.importance,
                "timestamp": m.timestamp
            }
            for m in self._recent_memories
        ]
    
    def _clear_memory_context(self) -> None:
        """
        Clear the current memory context.
        """
        self._recent_memories = [] 