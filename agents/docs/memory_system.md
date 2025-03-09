# Memory System Documentation

## Overview

The AgenticProcessGNN memory system enables agents to store, retrieve, and utilize past experiences, domain knowledge, 
and user preferences to improve their performance over time. This persistent memory capability allows agents to learn 
from past interactions and maintain context between sessions.

## System Architecture

The memory system is composed of several components that work together to provide a complete memory management solution:

```
src/memory/
├── __init__.py        # Package initialization
├── types.py           # Core memory data types
├── storage.py         # Storage implementations
├── manager.py         # Memory management interface
├── agent_memory.py    # Agent integration components
├── utils.py           # Utility functions
├── README.md          # Usage documentation
└── integration_example.py  # Example implementation
```

### Key Components

#### 1. Memory Types

The foundation of the memory system is the `Memory` class, which represents a single memory item. Each memory has the following attributes:

- **content**: The actual content of the memory (string)
- **memory_type**: The category of the memory (MemoryType enum)
- **memory_id**: A unique identifier for the memory (UUID)
- **metadata**: Additional contextual information (dictionary)
- **importance**: A value indicating the memory's significance (float between 0-1)
- **timestamp**: When the memory was created (datetime)
- **embedding**: Optional vector representation for similarity search (list of floats)

The `MemoryType` enum defines the following categories:
- **PROCESS**: Process designs and models
- **TASK**: Task-related information
- **USER_PREFERENCE**: User preferences and settings
- **INTERACTION**: Records of agent-user interactions
- **DOMAIN_KNOWLEDGE**: Domain-specific knowledge

#### 2. Storage Systems

Storage implementations provide persistence for memory objects. The system includes two storage options:

- **InMemoryStorage**: Volatile storage that keeps memories in RAM during runtime
- **FileStorage**: Persistent storage that saves memories to disk as JSON files

Both implementations share a common interface with the following operations:
- `add(memory)`: Store a new memory
- `get(memory_id)`: Retrieve a memory by ID
- `update(memory)`: Update an existing memory
- `delete(memory_id)`: Remove a memory
- `query(filter_func)`: Find memories based on custom criteria

#### 3. Memory Manager

The `MemoryManager` class provides a high-level interface for memory operations:

- **Adding memories**: Create and store new memories
- **Retrieving memories**: Get specific memories by ID
- **Querying memories**: Find memories based on type, content, or metadata
- **Updating memories**: Modify existing memories
- **Deleting memories**: Remove memories when no longer needed

#### 4. Agent Memory Integration

The `MemoryCapableMixin` class adds memory capabilities to agents:

- Initializes a memory manager for the agent
- Provides `remember()`, `recall()`, and `forget()` methods for memory operations
- Enables memory-based reasoning and decision making

## Memory Lifecycle

1. **Creation**: An agent creates a memory when it encounters important information
2. **Storage**: The memory is stored using the configured storage system
3. **Retrieval**: The agent recalls relevant memories when needed
4. **Utilization**: Retrieved memories influence agent behavior and decisions
5. **Update**: Memories can be updated with new information
6. **Deletion**: Outdated or unnecessary memories can be removed

## Integration with Agents

Agents can integrate memory capabilities by inheriting from the `MemoryCapableMixin` class:

```python
class MyAgent(BaseAgent, MemoryCapableMixin):
    def __init__(self, agent_id):
        BaseAgent.__init__(self, agent_id)
        MemoryCapableMixin.__init__(self)
        
    def process_task(self, task):
        # Recall relevant memories
        related_memories = self.recall(
            memory_type=MemoryType.PROCESS,
            metadata_filter={"domain": task.domain}
        )
        
        # Use memories to inform processing
        result = self._process_with_context(task, related_memories)
        
        # Remember the outcome
        self.remember(
            content=str(result),
            memory_type=MemoryType.TASK,
            metadata={"task_id": task.id, "outcome": "success"}
        )
        
        return result
```

## Performance Considerations

- **Storage Selection**: Choose the appropriate storage backend based on requirements:
  - `InMemoryStorage` for speed but lacks persistence
  - `FileStorage` for persistence but slower access

- **Memory Management**: Implement strategies to manage memory growth:
  - Prune less important memories
  - Archive old memories to secondary storage
  - Set expiration dates for temporary memories

- **Embedding Generation**: Vector embeddings enable semantic search but require additional processing:
  - Generate embeddings asynchronously for large memories
  - Cache embeddings to avoid redundant computation
  - Consider using a dedicated embedding service for large-scale systems

## Future Enhancements

Planned improvements to the memory system include:

1. **Database Storage**: Integration with SQL and document databases
2. **Vector Search**: Enhanced semantic similarity search using embeddings
3. **Memory Consolidation**: Mechanisms to combine related memories
4. **Memory Prioritization**: Improved algorithms for memory importance
5. **Distributed Memory**: Shared memory pools across agent instances

## Reference Implementation

For a complete example of how to use the memory system, see the `integration_example.py` file in the memory module. This example demonstrates how to enhance an existing agent with memory capabilities and showcases the benefits of persistent memory in a process design scenario. 