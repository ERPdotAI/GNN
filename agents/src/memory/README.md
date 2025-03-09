# Memory Module for AgenticProcessGNN

The Memory module provides a persistent memory system for agents in the AgenticProcessGNN framework, enabling them to remember and recall information across interactions and tasks.

## Overview

The Memory module implements a system inspired by Mem0, allowing agents to:

- Store different types of memories (process designs, tasks, domain knowledge, etc.)
- Retrieve memories based on various criteria
- Maintain context across interactions
- Track and use domain-specific knowledge
- Associate memories with specific agents

## Key Components

### 1. Memory Types

The module supports different types of memories:

- `PROCESS`: Process designs and analysis
- `TASK`: Task-specific information
- `USER_PREFERENCE`: User preferences and settings
- `INTERACTION`: Agent interactions and messages
- `DOMAIN_KNOWLEDGE`: Domain-specific knowledge

### 2. Storage Implementations

Two storage implementations are provided:

- **InMemoryStorage**: Fast, in-memory storage for development and testing
- **FileStorage**: Persistent, file-based storage for production use

### 3. Memory Manager

The `MemoryManager` provides high-level operations for working with memories:

- Creating memories of various types
- Retrieving memories by ID
- Searching for memories using filters
- Updating and deleting memories
- Specialized methods for each memory type

### 4. MemoryCapableMixin

The `MemoryCapableMixin` allows any agent to gain memory capabilities by:

- Adding it as a base class to your agent
- Initializing memory capabilities
- Using memory methods for remembering and recalling information

### 5. Utility Functions

Various utility functions to work with memories:

- Converting memories to text representations
- Calculating similarity between memories
- Applying time-based decay to memory importance
- Filtering and deduplicating memories

## Usage

### Basic Usage

```python
from memory import MemoryManager, MemoryType

# Create a memory manager
memory_manager = MemoryManager()

# Create a memory
memory_id = memory_manager.create_memory(
    content={"key": "value", "another_key": 123},
    memory_type=MemoryType.PROCESS,
    agent_id="agent-001",
    metadata={"domain": "banking"},
    importance=0.8
)

# Retrieve a memory
memory = memory_manager.get_memory(memory_id)

# Search memories
banking_memories = memory_manager.search_memories({
    "metadata": {"domain": "banking"}
})
```

### Adding Memory Capabilities to an Agent

```python
from memory import MemoryCapableMixin

class MyAgent(BaseAgent, MemoryCapableMixin):
    def __init__(self, agent_id):
        BaseAgent.__init__(self, agent_id)
        self._init_memory()  # Initialize memory capabilities
    
    def process_request(self, request):
        # Check memory for similar requests
        similar_requests = self._recall_memories({
            "metadata": {"request_type": request.type}
        })
        
        # Process the request
        result = self._process(request, similar_requests)
        
        # Remember this request and result
        self._remember(
            content={"request": request.to_dict(), "result": result.to_dict()},
            memory_type=MemoryType.TASK,
            metadata={"request_type": request.type}
        )
        
        return result
```

### Using File-Based Storage

```python
from memory import MemoryManager, FileStorage

# Create a file storage instance
file_storage = FileStorage(storage_dir="memory_data")

# Create a memory manager with file storage
memory_manager = MemoryManager(storage=file_storage)
```

## Advanced Usage

### Domain-Specific Memory Workflows

```python
# Store domain knowledge
memory_id = memory_manager.create_domain_knowledge_memory(
    agent_id="analyst-001",
    domain="banking",
    knowledge_type="regulation",
    content={
        "name": "KYC Compliance",
        "requirements": ["identity_verification", "address_verification"]
    }
)

# Retrieve domain knowledge
banking_regulations = memory_manager.get_domain_knowledge(
    domain="banking",
    knowledge_type="regulation"
)
```

### Memory Context

The mixin maintains a context of recent memories:

```python
# Get current memory context
context = agent._get_memory_context()

# Clear memory context
agent._clear_memory_context()
```

## Integration Example

See the `integration_example.py` file for a complete example showing how to enhance an existing agent with memory capabilities. The example demonstrates:

1. Creating an original agent without memory
2. Creating an enhanced agent with memory capabilities
3. Processing similar requests with both agents
4. Showing how the memory-enabled agent leverages previous knowledge

## Running Tests

Run the memory test script to verify functionality:

```bash
python -m src.memory.test_memory
```

## Developer Reference

- **types.py**: Core data types for memories
- **storage.py**: Storage implementations
- **manager.py**: High-level memory management
- **agent_memory.py**: Agent integration through mixin
- **utils.py**: Utility functions
- **test_memory.py**: Test script
- **integration_example.py**: Integration example 