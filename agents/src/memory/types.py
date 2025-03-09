"""
Memory types and core data structures.

This module defines the fundamental types and classes for the memory system,
including memory items, memory types, and serialization helpers.
"""

import uuid
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories that can be stored."""
    PROCESS = "process"               # Process designs and analysis
    TASK = "task"                     # Task-specific information 
    USER_PREFERENCE = "user_preference"  # User preferences and settings
    INTERACTION = "interaction"       # Agent interactions and messages
    DOMAIN_KNOWLEDGE = "domain_knowledge"  # Domain-specific knowledge

class Memory:
    """
    Represents a single memory item.
    """
    def __init__(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        timestamp: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ):
        """
        Initialize a new memory.
        
        Args:
            content: The content of the memory
            memory_type: The type of memory
            memory_id: Unique identifier for the memory (generated if not provided)
            metadata: Additional metadata about the memory
            importance: Importance score (0-1) for memory retrieval prioritization
            timestamp: When the memory was created (ISO format)
            embedding: Vector representation of the memory (if pre-computed)
        """
        self.memory_id = memory_id or str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.metadata = metadata or {}
        self.importance = importance
        self.timestamp = timestamp or datetime.datetime.utcnow().isoformat()
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary representation."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create a memory from dictionary representation."""
        return cls(
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            memory_id=data["memory_id"],
            metadata=data["metadata"],
            importance=data["importance"],
            timestamp=data["timestamp"],
            embedding=data.get("embedding")
        )
    
    def __repr__(self) -> str:
        """String representation of the memory."""
        return f"Memory(id={self.memory_id}, type={self.memory_type.value}, importance={self.importance})" 