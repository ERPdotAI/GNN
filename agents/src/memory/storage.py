"""
Memory storage systems.

This module provides storage implementations for the memory system,
including in-memory storage and disk-based persistence.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from pathlib import Path
import pickle
from datetime import datetime
from .types import Memory, MemoryType

# Setup logging
logger = logging.getLogger(__name__)

class MemoryStorage:
    """
    Abstract base class for memory storage systems.
    """
    def add(self, memory: Memory, agent_id: Optional[str] = None) -> str:
        """
        Add a memory to storage.
        
        Args:
            memory: The memory to store
            agent_id: Optional agent ID to associate with this memory
            
        Returns:
            The memory ID
        """
        raise NotImplementedError("Subclasses must implement add()")
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory object, or None if not found
        """
        raise NotImplementedError("Subclasses must implement get()")
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError("Subclasses must implement delete()")
    
    def update(self, memory_id: str, content: Optional[Dict[str, Any]] = None, 
              metadata: Optional[Dict[str, Any]] = None, 
              importance: Optional[float] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (optional)
            metadata: New metadata (optional)
            importance: New importance score (optional)
            
        Returns:
            True if updated, False if not found
        """
        raise NotImplementedError("Subclasses must implement update()")
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching the query.
        
        Args:
            query: Dictionary of search criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        raise NotImplementedError("Subclasses must implement search()")
    
    def get_agent_memories(self, agent_id: str) -> List[Memory]:
        """
        Get all memories associated with an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of memories
        """
        raise NotImplementedError("Subclasses must implement get_agent_memories()")
    
    def clear(self) -> None:
        """Clear all memories from storage."""
        raise NotImplementedError("Subclasses must implement clear()")

class InMemoryStorage(MemoryStorage):
    """
    In-memory implementation of memory storage.
    """
    def __init__(self, dimension: int = 512):
        """
        Initialize an in-memory storage system.
        
        Args:
            dimension: Dimension of memory embeddings
        """
        self.memories = {}  # memory_id -> Memory
        self.agent_memories = {}  # agent_id -> Set[memory_id]
        self.dimension = dimension
        logger.info("Initialized in-memory storage")
    
    def add(self, memory: Memory, agent_id: Optional[str] = None) -> str:
        """
        Add a memory to storage.
        
        Args:
            memory: The memory to store
            agent_id: Optional agent ID to associate with this memory
            
        Returns:
            The memory ID
        """
        # Store the memory
        self.memories[memory.memory_id] = memory
        
        # Associate with agent if provided
        if agent_id:
            if agent_id not in self.agent_memories:
                self.agent_memories[agent_id] = set()
            self.agent_memories[agent_id].add(memory.memory_id)
        
        logger.debug(f"Added memory {memory.memory_id} to storage")
        return memory.memory_id
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory object, or None if not found
        """
        return self.memories.get(memory_id)
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if memory_id not in self.memories:
            return False
        
        # Remove from main storage
        del self.memories[memory_id]
        
        # Remove from agent associations
        for agent_id, memory_ids in self.agent_memories.items():
            if memory_id in memory_ids:
                memory_ids.remove(memory_id)
        
        logger.debug(f"Deleted memory {memory_id}")
        return True
    
    def update(self, memory_id: str, content: Optional[Dict[str, Any]] = None, 
              metadata: Optional[Dict[str, Any]] = None, 
              importance: Optional[float] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (optional)
            metadata: New metadata (optional)
            importance: New importance score (optional)
            
        Returns:
            True if updated, False if not found
        """
        if memory_id not in self.memories:
            return False
        
        memory = self.memories[memory_id]
        
        if content is not None:
            memory.content = content
        
        if metadata is not None:
            memory.metadata = metadata
        
        if importance is not None:
            memory.importance = importance
        
        logger.debug(f"Updated memory {memory_id}")
        return True
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching the query.
        
        Args:
            query: Dictionary of search criteria (supports memory_type, agent_id, metadata fields)
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Filter by memory type if specified
        memory_type = query.get("memory_type")
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                logger.warning(f"Invalid memory_type in query: {memory_type}")
                return []
        
        # Filter by agent ID if specified
        agent_id = query.get("agent_id")
        if agent_id and agent_id in self.agent_memories:
            memory_ids = self.agent_memories[agent_id]
            candidates = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            candidates = list(self.memories.values())
        
        # Apply memory type filter
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        # Apply metadata filters
        metadata_filters = query.get("metadata", {})
        if metadata_filters:
            for key, value in metadata_filters.items():
                candidates = [m for m in candidates if m.metadata.get(key) == value]
        
        # Sort by importance and return top results
        results = sorted(candidates, key=lambda m: m.importance, reverse=True)[:limit]
        return results
    
    def get_agent_memories(self, agent_id: str) -> List[Memory]:
        """
        Get all memories associated with an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of memories
        """
        if agent_id not in self.agent_memories:
            return []
        
        memory_ids = self.agent_memories[agent_id]
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]
    
    def clear(self) -> None:
        """Clear all memories from storage."""
        self.memories.clear()
        self.agent_memories.clear()
        logger.debug("Cleared all memories from storage")

class FileStorage(MemoryStorage):
    """
    File-based implementation of memory storage.
    """
    def __init__(self, storage_dir: str = "memory_data"):
        """
        Initialize a file-based storage system.
        
        Args:
            storage_dir: Directory to store memory data
        """
        self.storage_dir = Path(storage_dir)
        self.memories_dir = self.storage_dir / "memories"
        self.index_path = self.storage_dir / "index.json"
        self.agent_index_path = self.storage_dir / "agent_index.json"
        
        # Create directories if they don't exist
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize indexes
        self._load_indexes()
        
        logger.info(f"Initialized file-based storage in {storage_dir}")
    
    def _load_indexes(self) -> None:
        """Load or initialize the memory indexes."""
        # Memory index maps memory_id -> metadata for quick lookup
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.memory_index = json.load(f)
        else:
            self.memory_index = {}
            self._save_memory_index()
        
        # Agent index maps agent_id -> list of memory_ids
        if self.agent_index_path.exists():
            with open(self.agent_index_path, 'r') as f:
                self.agent_index = json.load(f)
        else:
            self.agent_index = {}
            self._save_agent_index()
    
    def _save_memory_index(self) -> None:
        """Save the memory index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump(self.memory_index, f, indent=2)
    
    def _save_agent_index(self) -> None:
        """Save the agent index to disk."""
        with open(self.agent_index_path, 'w') as f:
            json.dump(self.agent_index, f, indent=2)
    
    def _memory_path(self, memory_id: str) -> Path:
        """Get the path to a memory file."""
        return self.memories_dir / f"{memory_id}.json"
    
    def add(self, memory: Memory, agent_id: Optional[str] = None) -> str:
        """
        Add a memory to storage.
        
        Args:
            memory: The memory to store
            agent_id: Optional agent ID to associate with this memory
            
        Returns:
            The memory ID
        """
        memory_id = memory.memory_id
        memory_path = self._memory_path(memory_id)
        
        # Store the memory data
        with open(memory_path, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
        
        # Update the memory index
        self.memory_index[memory_id] = {
            "type": memory.memory_type.value,
            "timestamp": memory.timestamp,
            "importance": memory.importance
        }
        self._save_memory_index()
        
        # Associate with agent if provided
        if agent_id:
            if agent_id not in self.agent_index:
                self.agent_index[agent_id] = []
            if memory_id not in self.agent_index[agent_id]:
                self.agent_index[agent_id].append(memory_id)
                self._save_agent_index()
        
        logger.debug(f"Added memory {memory_id} to file storage")
        return memory_id
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a memory by ID.
        
        Args:
            memory_id: ID of the memory to retrieve
            
        Returns:
            The memory object, or None if not found
        """
        memory_path = self._memory_path(memory_id)
        
        if not memory_path.exists():
            return None
        
        try:
            with open(memory_path, 'r') as f:
                memory_data = json.load(f)
                return Memory.from_dict(memory_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading memory {memory_id}: {str(e)}")
            return None
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        memory_path = self._memory_path(memory_id)
        
        if not memory_path.exists():
            return False
        
        # Delete the memory file
        memory_path.unlink()
        
        # Update the memory index
        if memory_id in self.memory_index:
            del self.memory_index[memory_id]
            self._save_memory_index()
        
        # Update agent associations
        for agent_id, memory_ids in self.agent_index.items():
            if memory_id in memory_ids:
                self.agent_index[agent_id].remove(memory_id)
        self._save_agent_index()
        
        logger.debug(f"Deleted memory {memory_id} from file storage")
        return True
    
    def update(self, memory_id: str, content: Optional[Dict[str, Any]] = None, 
              metadata: Optional[Dict[str, Any]] = None, 
              importance: Optional[float] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: ID of the memory to update
            content: New content (optional)
            metadata: New metadata (optional)
            importance: New importance score (optional)
            
        Returns:
            True if updated, False if not found
        """
        memory = self.get(memory_id)
        
        if memory is None:
            return False
        
        if content is not None:
            memory.content = content
        
        if metadata is not None:
            memory.metadata = metadata
        
        if importance is not None:
            memory.importance = importance
            # Update the index with new importance
            self.memory_index[memory_id]["importance"] = importance
            self._save_memory_index()
        
        # Save the updated memory
        memory_path = self._memory_path(memory_id)
        with open(memory_path, 'w') as f:
            json.dump(memory.to_dict(), f, indent=2)
        
        logger.debug(f"Updated memory {memory_id} in file storage")
        return True
    
    def search(self, query: Dict[str, Any], limit: int = 10) -> List[Memory]:
        """
        Search for memories matching the query.
        
        Args:
            query: Dictionary of search criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        # First, filter memory IDs by the index
        memory_ids = set(self.memory_index.keys())
        
        # Filter by memory type if specified
        memory_type = query.get("memory_type")
        if memory_type:
            if isinstance(memory_type, str):
                memory_type_value = memory_type
            else:
                memory_type_value = memory_type.value
            
            memory_ids = {
                mid for mid in memory_ids 
                if self.memory_index[mid]["type"] == memory_type_value
            }
        
        # Filter by agent ID if specified
        agent_id = query.get("agent_id")
        if agent_id:
            if agent_id in self.agent_index:
                agent_memory_ids = set(self.agent_index[agent_id])
                memory_ids &= agent_memory_ids
            else:
                return []  # No memories for this agent
        
        # Load memories for detailed filtering and sorting
        memories = []
        for mid in memory_ids:
            memory = self.get(mid)
            if memory:
                # Apply metadata filters
                metadata_filters = query.get("metadata", {})
                if metadata_filters:
                    match = True
                    for key, value in metadata_filters.items():
                        if memory.metadata.get(key) != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                memories.append(memory)
        
        # Sort by importance and return top results
        return sorted(memories, key=lambda m: m.importance, reverse=True)[:limit]
    
    def get_agent_memories(self, agent_id: str) -> List[Memory]:
        """
        Get all memories associated with an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of memories
        """
        if agent_id not in self.agent_index:
            return []
        
        memories = []
        for memory_id in self.agent_index[agent_id]:
            memory = self.get(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def clear(self) -> None:
        """Clear all memories from storage."""
        # Remove all memory files
        for memory_file in self.memories_dir.glob("*.json"):
            memory_file.unlink()
        
        # Reset indexes
        self.memory_index = {}
        self.agent_index = {}
        self._save_memory_index()
        self._save_agent_index()
        
        logger.debug("Cleared all memories from file storage") 