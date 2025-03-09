"""
Memory utilities.

This module provides utility functions for working with memories,
including embedding generation and similarity calculations.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import time
from datetime import datetime, timedelta
import pytz

from .types import Memory, MemoryType

# Setup logging
logger = logging.getLogger(__name__)

def format_memory_timestamp(dt: datetime) -> str:
    """
    Format a datetime object to ISO 8601 format with timezone information.
    
    Args:
        dt: Datetime object to format
        
    Returns:
        Formatted timestamp string
    """
    # Ensure datetime is timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    
    # Format to ISO 8601 with timezone info
    return dt.isoformat()

def parse_memory_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO 8601 formatted timestamp string to a datetime object.
    
    Args:
        timestamp_str: ISO 8601 formatted timestamp string
        
    Returns:
        Datetime object
    """
    # Handle 'Z' notation for UTC
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1] + '+00:00'
    
    # Parse the timestamp
    dt = datetime.fromisoformat(timestamp_str)
    
    # Ensure timezone is set
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=pytz.UTC)
    
    return dt

def memory_to_text(memory: Memory) -> str:
    """
    Convert a memory to a text representation.
    
    Args:
        memory: Memory to convert
        
    Returns:
        Text representation of the memory
    """
    try:
        # Format content and metadata as JSON strings
        content_str = json.dumps(memory.content, indent=2)
        metadata_str = json.dumps(memory.metadata, indent=2)
        
        return f"""Memory ID: {memory.memory_id}
Type: {memory.memory_type.value}
Time: {memory.timestamp}
Importance: {memory.importance}
Content: {content_str}
Metadata: {metadata_str}
"""
    except Exception as e:
        logger.error(f"Error converting memory to text: {str(e)}")
        return f"Memory {memory.memory_id} (error in conversion)"

def memories_to_text(memories: List[Memory]) -> str:
    """
    Convert a list of memories to a text representation.
    
    Args:
        memories: List of memories to convert
        
    Returns:
        Text representation of the memories
    """
    if not memories:
        return "No memories."
    
    result = f"Memory Summary ({len(memories)} items):\n"
    for i, memory in enumerate(memories):
        result += f"\n--- Memory {i+1} ---\n"
        result += memory_to_text(memory)
    
    return result

def format_memory_for_agent(memory: Memory) -> Dict[str, Any]:
    """
    Format a memory for use by an agent in a more structured way.
    
    Args:
        memory: Memory to format
        
    Returns:
        Dictionary representation of the memory
    """
    return {
        "id": memory.memory_id,
        "type": memory.memory_type.value,
        "timestamp": memory.timestamp,
        "content": memory.content,
        "metadata": memory.metadata,
        "importance": memory.importance
    }

def importance_decay(memories: List[Memory], days_half_life: float = 30.0) -> List[Memory]:
    """
    Apply time-based importance decay to memories.
    
    Args:
        memories: List of memories to decay
        days_half_life: Number of days for importance to halve
        
    Returns:
        List of memories with decayed importance
    """
    if not memories:
        return []
    
    now = datetime.utcnow()
    decayed_memories = []
    
    for memory in memories:
        try:
            # Parse timestamp
            if isinstance(memory.timestamp, str):
                timestamp = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
            else:
                timestamp = memory.timestamp
            
            # Calculate age in days
            age_td = now - timestamp
            age_days = age_td.total_seconds() / (60 * 60 * 24)
            
            # Apply exponential decay
            decay_factor = 2 ** (-age_days / days_half_life)
            decayed_importance = memory.importance * decay_factor
            
            # Create a new memory with decayed importance
            decayed_memory = Memory(
                content=memory.content,
                memory_type=memory.memory_type,
                memory_id=memory.memory_id,
                metadata=memory.metadata,
                importance=decayed_importance,
                timestamp=memory.timestamp,
                embedding=memory.embedding
            )
            
            decayed_memories.append(decayed_memory)
        except Exception as e:
            logger.error(f"Error decaying memory {memory.memory_id}: {str(e)}")
            decayed_memories.append(memory)
    
    return decayed_memories

def filter_memories_by_recency(memories: List[Memory], days: int = 30) -> List[Memory]:
    """
    Filter memories by recency.
    
    Args:
        memories: List of memories to filter
        days: Maximum age in days
        
    Returns:
        List of recent memories
    """
    if not memories:
        return []
    
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)
    recent_memories = []
    
    for memory in memories:
        try:
            # Parse timestamp
            if isinstance(memory.timestamp, str):
                timestamp = datetime.fromisoformat(memory.timestamp.replace('Z', '+00:00'))
            else:
                timestamp = memory.timestamp
            
            if timestamp >= cutoff:
                recent_memories.append(memory)
        except Exception as e:
            logger.error(f"Error filtering memory {memory.memory_id}: {str(e)}")
    
    return recent_memories

def memory_similarity(memory1: Memory, memory2: Memory) -> float:
    """
    Calculate similarity between two memories.
    
    Args:
        memory1: First memory
        memory2: Second memory
        
    Returns:
        Similarity score (0.0-1.0)
    """
    if memory1.embedding is None or memory2.embedding is None:
        # Fall back to content-based similarity if embeddings not available
        return content_similarity(memory1.content, memory2.content)
    
    # Convert to numpy arrays
    embedding1 = np.array(memory1.embedding)
    embedding2 = np.array(memory2.embedding)
    
    # Normalize vectors
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2
    
    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2)
    
    # Ensure value is in [0, 1] range
    return max(0.0, min(1.0, similarity))

def content_similarity(content1: Dict[str, Any], content2: Dict[str, Any]) -> float:
    """
    Calculate similarity between two content dictionaries.
    
    Args:
        content1: First content dictionary
        content2: Second content dictionary
        
    Returns:
        Similarity score (0.0-1.0)
    """
    # Simple implementation - compare keys
    keys1 = set(content1.keys())
    keys2 = set(content2.keys())
    
    if not keys1 or not keys2:
        return 0.0
    
    # Jaccard similarity of keys
    intersection = len(keys1.intersection(keys2))
    union = len(keys1.union(keys2))
    
    key_similarity = intersection / union if union > 0 else 0.0
    
    # Check values for common keys
    value_similarity = 0.0
    common_keys = keys1.intersection(keys2)
    
    if common_keys:
        matches = 0
        for key in common_keys:
            if content1[key] == content2[key]:
                matches += 1
        
        value_similarity = matches / len(common_keys)
    
    # Combine key and value similarity
    return 0.5 * key_similarity + 0.5 * value_similarity

def deduplicate_memories(memories: List[Memory], threshold: float = 0.9) -> List[Memory]:
    """
    Remove duplicate memories.
    
    Args:
        memories: List of memories to deduplicate
        threshold: Similarity threshold for considering memories as duplicates
        
    Returns:
        Deduplicated list of memories
    """
    if not memories:
        return []
    
    # Sort by importance (high to low)
    sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)
    
    # Initialize result with the most important memory
    result = [sorted_memories[0]]
    
    # Check each memory against the result
    for memory in sorted_memories[1:]:
        is_duplicate = False
        
        for existing in result:
            similarity = memory_similarity(memory, existing)
            if similarity >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.append(memory)
    
    return result 