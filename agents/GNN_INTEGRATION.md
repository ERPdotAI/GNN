# Integration of Agent Framework with GNN Process Mining

This directory contains the ERP•AI Agent Framework, which builds upon the Graph Neural Network (GNN) process mining capabilities found in the parent directory. The agent framework leverages the GNN models to create intelligent, autonomous agents that can understand and optimize business processes.

## Relationship to GNN Process Mining

The GNN Process Mining project provides the foundational models and algorithms for understanding business processes through graph-based representations. The Agent Framework extends this by:

1. Creating autonomous agents that can act upon the process insights
2. Providing a memory system for persistent agent knowledge
3. Enabling API-based interaction with the process models
4. Supporting integration with enterprise systems
5. Offering a vector database for semantic search capabilities

## Key Components

- **Agent Framework** (`src/agents/`): Core agent implementation with decision-making capabilities
- **Memory System** (`src/memory/`): Long-term and short-term memory storage for agents
- **Process Engine** (`src/process_engine/`): Integration with the GNN process mining models
- **API** (`src/api/`): RESTful interfaces for interacting with agents and processes
- **Vector Database** (`src/vector_db/`): Semantic search capabilities for process knowledge

## Usage with GNN Process Mining

To use the Agent Framework with the GNN Process Mining capabilities:

1. Train process models using the GNN implementation in the parent directory
2. Configure agents to use these models via the Process Engine integration
3. Deploy agents that can monitor, analyze, and optimize processes in real-time

## Example Integration Flow

```python
# Import from parent GNN directory
from models.gat_model import NextTaskGAT

# Import from agents framework
from agents.src.process_engine.gnn_model import ProcessGNNModel
from agents.src.agents.agent_framework import ProcessAgent

# Load trained GNN model
gnn_model = NextTaskGAT(...)
gnn_model.load_state_dict(torch.load("models/best_gnn_model.pth"))

# Create process model wrapper for agent
process_model = ProcessGNNModel(gnn_model)

# Initialize agent with process model
agent = ProcessAgent(
    name="inventory_optimization_agent",
    process_model=process_model,
    objective="Optimize inventory levels and reorder timing"
)

# Deploy agent to monitor and optimize processes
agent.deploy()
```

This integration shows how the comprehensive process mining capabilities of the main GNN project can be leveraged through intelligent agents to create autonomous business process optimization. 