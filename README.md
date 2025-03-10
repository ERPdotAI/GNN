# ERP•AI Process GNN Framework

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-2.0.0-green.svg)](https://github.com/ERPdotAI/GNN/tree/AdvancedGNN)

## Process Mining with Graph Neural Networks and Agent Framework

The AdvancedGNN branch extends our original GNN-based process mining framework with activity group semantics, enhanced neural architectures, and an integrated agent framework for process automation and optimization. This research implementation combines state-of-the-art machine learning models with traditional process mining methods.

**Analyze, predict, and optimize business processes with advanced AI techniques.**

## 🔍 Key Features

### 🧠 Enhanced Neural Architecture

The AdvancedGNN branch includes significant architectural improvements:

- **Activity Group Embeddings** for semantic understanding of process relationships
- **Layer Normalization and Residual Connections** for improved training stability
- **GELU Activation Functions** replacing traditional ReLU activations
- **Early Stopping with Patience** for more efficient training
- **Group-aware Attention Mechanisms** for better context understanding

### 📊 Process Mining Components

Core process mining capabilities implemented in the code:

- **Bottleneck Detection** using temporal analysis between activities
- **Conformance Checking** with inductive miner implementation
- **Cycle Time Analysis** for process duration insights
- **Transition Pattern Discovery** for process flow understanding
- **Spectral Clustering** for process segmentation

### 🤖 Agent Framework Integration

The integrated agent framework provides:

- **Process Agents** that can monitor and act on GNN insights
- **Memory System** with persistent storage for agent knowledge
- **Process Engine Integration** with GNN model connections
- **Vector Database** for semantic search capabilities
- **API Endpoints** for agent and process interactions

## 📁 Technical Architecture

```
src/
├── input/                # input files
├── models/
│   ├── gat_model.py      # Enhanced Graph Attention Network 
│   └── lstm_model.py     # LSTM sequence model
├── modules/
│   ├── data_preprocessing.py  # Data handling and feature engineering
│   ├── process_mining.py     # Core process mining functions
│   └── rl_optimization.py    # Reinforcement learning components
├── visualization/
│   └── process_viz.py        # Visualization toolkit
├── activity_groups.py        # Activity group definitions
├── agents/                   # Autonomous agent framework
│   ├── src/
│   │   ├── agents/           # Agent implementation
│   │   ├── memory/           # Agent memory system
│   │   ├── process_engine/   # Process model integration
│   │   ├── api/              # API endpoints
│   │   └── vector_db/        # Vector database implementation
│   ├── GNN_INTEGRATION.md    # Documentation on integration with GNN
│   └── requirements.txt      # Agent-specific dependencies
└── main.py                   # Main execution script
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- PM4Py
- NetworkX
- Additional dependencies in requirements.txt

### Installation

```bash
# Clone the repository
git clone https://github.com/ERPdotAI/GNN.git
cd GNN

# Switch to AdvancedGNN branch
git checkout AdvancedGNN

# Install dependencies for GNN process mining
pip install -r requirements.txt

# For the agent framework (optional)
cd agents
pip install -r requirements.txt
```

### Running Process Mining

```bash
# Basic usage
python main.py --data_path <input-file-path>

# Advanced usage with all options
python main.py --data_path <input-file-path> --sample_size 2000 --batch_size 32 \
               --num_epochs 100 --learning_rate 0.001 --hidden_dim 64 \
               --num_layers 2 --heads 4 --dropout 0.5 --use_norm_features
```

### Using the Agent Framework

```bash
cd agents
python src/main.py
```

## 📋 Implementation Details

The code in this branch implements the following:

### GNN Implementation
- **Graph Attention Networks** with multi-head attention mechanisms
- **Activity Group Embeddings** integrated into the GAT architecture
- **Improved Sequence Modeling** with enhanced LSTM implementation
- **Advanced Preprocessing** with semantic feature engineering

### Process Mining Components
- **Bottleneck Analysis** that identifies process inefficiencies
- **Cycle Time Analysis** for understanding process durations
- **Conformance Checking** against reference process models
- **Transition Analysis** for understanding process flows
- **Spectral Clustering** for process segmentation

### Agent Framework
- **Agent Framework Core** for implementing autonomous process agents
- **Memory System** for persistent agent knowledge
- **Process Engine Integration** with GNN models
- **Vector Database** for semantic search of process knowledge
- **API Layer** for interaction with agents and processes

## 🔄 GNN-Agent Integration

The AdvancedGNN branch introduces a unique integration between GNN-based process mining and autonomous agents:

```python
# Example integration flow
from models.gat_model import NextTaskGAT
from agents.src.process_engine.gnn_model import ProcessGNNModel
from agents.src.agents.agent_framework import ProcessAgent

# Load trained GNN model
gnn_model = NextTaskGAT(...)
gnn_model.load_state_dict(torch.load("models/best_gnn_model.pth"))

# Create process model wrapper for agent
process_model = ProcessGNNModel(gnn_model)

# Initialize agent with process model
agent = ProcessAgent(
    name="process_optimization_agent",
    process_model=process_model,
    objective="Optimize process efficiency"
)

# Deploy agent to monitor and optimize processes
agent.deploy()
```

See `agents/GNN_INTEGRATION.md` for detailed integration documentation.

## 📊 Implemented Use Cases

The framework currently supports analysis and optimization for:

- **Process Bottleneck Identification**
- **Next Activity Prediction**
- **Process Conformance Analysis**
- **Activity Clustering**
- **Process Flow Visualization**

## 📚 Documentation

The architecture is documented in the following files:
- `README.md` - Overview and getting started
- `ARCHITECTURE.md` - Detailed architecture diagrams
- `agents/GNN_INTEGRATION.md` - Agent integration documentation

## 🤝 Contributing

We welcome contributions from the research community. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request with detailed documentation

## 📄 License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## 🔬 Research Context

This is a research implementation combining process mining with graph neural networks and agent systems. The code demonstrates a sophisticated approach to business process analysis and automation.

---

<p align="center">© 2025 ERP•AI. All rights reserved.</p> 
