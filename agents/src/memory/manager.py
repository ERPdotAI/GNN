"""
Memory management system.

This module provides the main interface for agents to interact with memories,
including creation, retrieval, and search operations.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid

from .types import Memory, MemoryType
from .storage import MemoryStorage, InMemoryStorage, FileStorage

# Setup logging
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory management system that provides the main interface for agents to interact with memories.
    """
    def __init__(self, storage: Optional[MemoryStorage] = None, use_embeddings: bool = True):
        """
        Initialize the memory manager.
        
        Args:
            storage: Storage implementation to use (defaults to InMemoryStorage)
            use_embeddings: Whether to use embeddings for semantic search
        """
        self.storage = storage or InMemoryStorage()
        self.use_embeddings = use_embeddings
        logger.info("Initialized memory manager")
    
    def create_memory(self, content: Dict[str, Any], memory_type: MemoryType, 
                     agent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                     importance: float = 0.5, embedding: Optional[List[float]] = None) -> str:
        """
        Create a new memory.
        
        Args:
            content: The content of the memory
            memory_type: Type of memory
            agent_id: ID of the agent creating the memory (optional)
            metadata: Additional metadata for the memory (optional)
            importance: Importance score (0.0-1.0)
            embedding: Pre-computed embedding (optional)
            
        Returns:
            The ID of the created memory
        """
        # Create the memory
        memory = Memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata or {},
            importance=importance,
            embedding=embedding
        )
        
        # Store the memory
        memory_id = self.storage.add(memory, agent_id)
        logger.debug(f"Created memory {memory_id} of type {memory_type.value}")
        
        return memory_id
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory, or None if not found
        """
        return self.storage.get(memory_id)
    
    def update_memory(self, memory_id: str, content: Optional[Dict[str, Any]] = None,
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
        return self.storage.update(memory_id, content, metadata, importance)
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        return self.storage.delete(memory_id)
    
    def search_memories(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching the query.
        
        Args:
            query: Dictionary of search criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        return self.storage.search(query, limit)
    
    def get_agent_memories(self, agent_id: str, memory_type: Optional[MemoryType] = None,
                         limit: int = 10) -> List[Memory]:
        """
        Get memories for an agent, optionally filtered by type.
        
        Args:
            agent_id: ID of the agent
            memory_type: Optional type filter
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        query = {"agent_id": agent_id}
        if memory_type:
            query["memory_type"] = memory_type
        
        return self.storage.search(query, limit)
    
    def clear_memories(self) -> None:
        """
        Clear all memories from storage.
        """
        self.storage.clear()
        logger.debug("Cleared all memories")
    
    # Process-specific methods
    def create_process_memory(self, agent_id: str, process_id: str, process_name: str,
                           content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory related to a process.
        
        Args:
            agent_id: ID of the agent creating the memory
            process_id: ID of the process
            process_name: Name of the process
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        metadata = {
            "process_id": process_id,
            "process_name": process_name
        }
        
        return self.create_memory(
            content=content,
            memory_type=MemoryType.PROCESS,
            agent_id=agent_id,
            metadata=metadata,
            importance=importance
        )
    
    def get_process_memories(self, process_id: str, limit: int = 10) -> List[Memory]:
        """
        Get memories related to a specific process.
        
        Args:
            process_id: ID of the process
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        query = {
            "memory_type": MemoryType.PROCESS,
            "metadata": {"process_id": process_id}
        }
        
        return self.storage.search(query, limit)
    
    # Task-specific methods
    def create_task_memory(self, agent_id: str, task_id: str, task_name: str,
                        content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory related to a task.
        
        Args:
            agent_id: ID of the agent creating the memory
            task_id: ID of the task
            task_name: Name of the task
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        metadata = {
            "task_id": task_id,
            "task_name": task_name
        }
        
        return self.create_memory(
            content=content,
            memory_type=MemoryType.TASK,
            agent_id=agent_id,
            metadata=metadata,
            importance=importance
        )
    
    def get_task_memories(self, task_id: str, limit: int = 10) -> List[Memory]:
        """
        Get memories related to a specific task.
        
        Args:
            task_id: ID of the task
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        query = {
            "memory_type": MemoryType.TASK,
            "metadata": {"task_id": task_id}
        }
        
        return self.storage.search(query, limit)
    
    # User preference-specific methods
    def create_user_preference_memory(self, agent_id: str, user_id: str, preference_type: str,
                                  content: Dict[str, Any], importance: float = 0.7) -> str:
        """
        Create a memory about a user preference.
        
        Args:
            agent_id: ID of the agent creating the memory
            user_id: ID of the user
            preference_type: Type of preference (e.g., "workflow_style", "ui_preference")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        metadata = {
            "user_id": user_id,
            "preference_type": preference_type
        }
        
        return self.create_memory(
            content=content,
            memory_type=MemoryType.USER_PREFERENCE,
            agent_id=agent_id,
            metadata=metadata,
            importance=importance
        )
    
    def get_user_preferences(self, user_id: str, preference_type: Optional[str] = None,
                           limit: int = 10) -> List[Memory]:
        """
        Get memories about user preferences.
        
        Args:
            user_id: ID of the user
            preference_type: Optional type of preference to filter by
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        metadata = {"user_id": user_id}
        if preference_type:
            metadata["preference_type"] = preference_type
        
        query = {
            "memory_type": MemoryType.USER_PREFERENCE,
            "metadata": metadata
        }
        
        return self.storage.search(query, limit)
    
    # Interaction memory methods
    def create_interaction_memory(self, agent_id: str, user_id: str, interaction_type: str,
                               content: Dict[str, Any], importance: float = 0.5) -> str:
        """
        Create a memory about an interaction.
        
        Args:
            agent_id: ID of the agent creating the memory
            user_id: ID of the user
            interaction_type: Type of interaction (e.g., "feedback", "request")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        metadata = {
            "user_id": user_id,
            "interaction_type": interaction_type
        }
        
        return self.create_memory(
            content=content,
            memory_type=MemoryType.INTERACTION,
            agent_id=agent_id,
            metadata=metadata,
            importance=importance
        )
    
    def get_interactions(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                       interaction_type: Optional[str] = None, limit: int = 10) -> List[Memory]:
        """
        Get memories about interactions.
        
        Args:
            user_id: Optional ID of the user to filter by
            agent_id: Optional ID of the agent to filter by
            interaction_type: Optional type of interaction to filter by
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        metadata = {}
        if user_id:
            metadata["user_id"] = user_id
        
        if interaction_type:
            metadata["interaction_type"] = interaction_type
        
        query = {
            "memory_type": MemoryType.INTERACTION,
            "metadata": metadata
        }
        
        if agent_id:
            query["agent_id"] = agent_id
        
        return self.storage.search(query, limit)
    
    # Domain knowledge methods
    def create_domain_knowledge_memory(self, agent_id: str, domain: str, knowledge_type: str,
                                    content: Dict[str, Any], importance: float = 0.8) -> str:
        """
        Create a memory about domain knowledge.
        
        Args:
            agent_id: ID of the agent creating the memory
            domain: Domain name (e.g., "banking", "healthcare")
            knowledge_type: Type of knowledge (e.g., "best_practice", "regulation")
            content: Content of the memory
            importance: Importance score
            
        Returns:
            The ID of the created memory
        """
        metadata = {
            "domain": domain,
            "knowledge_type": knowledge_type
        }
        
        return self.create_memory(
            content=content,
            memory_type=MemoryType.DOMAIN_KNOWLEDGE,
            agent_id=agent_id,
            metadata=metadata,
            importance=importance
        )
    
    def get_domain_knowledge(self, domain: Optional[str] = None, 
                           knowledge_type: Optional[str] = None,
                           limit: int = 10) -> List[Memory]:
        """
        Get memories about domain knowledge.
        
        Args:
            domain: Optional domain to filter by
            knowledge_type: Optional type of knowledge to filter by
            limit: Maximum number of results
            
        Returns:
            List of memories
        """
        metadata = {}
        if domain:
            metadata["domain"] = domain
        
        if knowledge_type:
            metadata["knowledge_type"] = knowledge_type
        
        query = {
            "memory_type": MemoryType.DOMAIN_KNOWLEDGE,
            "metadata": metadata
        }
        
        return self.storage.search(query, limit)
    
    def summarize_memories(self, memories: List[Memory]) -> Dict[str, Any]:
        """
        Generate a summary of a list of memories.
        
        Args:
            memories: List of memories to summarize
            
        Returns:
            Summary information as a dictionary
        """
        if not memories:
            return {
                "count": 0,
                "types": {},
                "average_importance": 0.0,
                "oldest": None,
                "newest": None
            }
        
        # Count by type
        types = {}
        for memory in memories:
            type_value = memory.memory_type.value
            types[type_value] = types.get(type_value, 0) + 1
        
        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Find oldest and newest
        timestamps = [m.timestamp for m in memories]
        oldest = min(timestamps)
        newest = max(timestamps)
        
        return {
            "count": len(memories),
            "types": types,
            "average_importance": avg_importance,
            "oldest": oldest,
            "newest": newest
        } 