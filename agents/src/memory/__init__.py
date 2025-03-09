"""
Memory system for the Agentic Process GNN.

This package provides persistent memory capabilities for agents,
enabling them to remember task-specific information across interactions.
"""

from .types import Memory, MemoryType
from .storage import MemoryStorage, InMemoryStorage, FileStorage
from .manager import MemoryManager
from .agent_memory import MemoryCapableMixin
from .utils import (
    memory_to_text, 
    memories_to_text, 
    format_memory_for_agent,
    importance_decay,
    filter_memories_by_recency,
    memory_similarity,
    content_similarity,
    deduplicate_memories
)

__all__ = [
    'Memory',
    'MemoryType',
    'MemoryStorage',
    'InMemoryStorage',
    'FileStorage',
    'MemoryManager',
    'MemoryCapableMixin',
    'memory_to_text',
    'memories_to_text',
    'format_memory_for_agent',
    'importance_decay',
    'filter_memories_by_recency',
    'memory_similarity',
    'content_similarity',
    'deduplicate_memories'
]
