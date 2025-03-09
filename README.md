## Process Mining with Graph Neural Networks

An advanced implementation combining Graph Neural Networks, Deep Learning, and Process Mining techniques for business process analysis and prediction.

## 1. Overview

This research project implements a novel approach to process mining using Graph Neural Networks (GNN) and deep learning techniques. The framework combines state-of-the-art machine learning models with traditional process mining methods to provide comprehensive process analysis and prediction capabilities.

> **New in AdvancedGNN Branch:** This branch includes significant enhancements including activity groups for semantic modeling, improved neural architectures, enhanced training procedures, and an integrated agent framework for process automation.

## 2. Authors

- **Somesh Misra** [@mathprobro](https://x.com/mathprobro)
- **Shashank Dixit** [@sphinx](https://x.com/protosphinx)
- **Research Group**: [ERP.AI](https://www.erp.ai) Research

## 3. Key Components

1. **Process Analysis**
- Advanced bottleneck detection using temporal analysis
- Conformance checking with inductive mining
- Cycle time analysis and prediction
- Transition pattern discovery
- Spectral clustering for process segmentation
- **New:** Activity groups for semantic categorization of business processes
- **New:** Enhanced cluster analysis with statistical insights

2. **Machine Learning Models**
- Graph Attention Networks (GAT) for structural learning
- LSTM networks for temporal dependencies
- Reinforcement Learning for process optimization
- Custom neural architectures for process prediction
- **New:** Group-aware attention mechanisms
- **New:** Layer normalization and residual connections
- **New:** GELU activation functions
- **New:** Improved gradient flow with better training stability

3. **Visualization Suite**
- Interactive process flow visualization
- Temporal pattern analysis
- Performance bottleneck identification
- Resource utilization patterns
- Custom process metrics
- **New:** Enhanced experiment documentation

4. **Agent Framework** (New in AdvancedGNN)
- Autonomous process agents that act on insights
- Memory system for persistent knowledge
- API-based interaction with process models
- Enterprise system integration capabilities
- Vector database for semantic search

## 4. Technical Architecture

```
src/
├── input/                # input files
├── models/
│   ├── gat_model.py      # Graph Attention Network implementation
│   └── lstm_model.py     # LSTM sequence model
├── modules/
│   ├── data_preprocessing.py  # Data handling and feature engineering
│   ├── process_mining.py     # Core process mining functions
│   └── rl_optimization.py    # Reinforcement learning components
├── visualization/
│   └── process_viz.py        # Visualization toolkit
├── activity_groups.py        # Activity group definitions
├── agents/                   # Autonomous agent framework (New)
│   ├── src/
│   │   ├── agents/           # Agent implementation
│   │   ├── memory/           # Agent memory system
│   │   ├── process_engine/   # Process model integration
│   │   ├── api/              # API endpoints
│   │   └── vector_db/        # Vector database implementation
└── main.py                   # Main execution script
```

## 5. Technical Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- PM4Py
- NetworkX
- Additional dependencies in requirements.txt

## 6. Installation

1. Clone the repository:
```bash
git clone https://github.com/ERPdotAI/GNN.git
cd GNN
```

2. Switch to the AdvancedGNN branch (for enhanced features):
```bash
git checkout AdvancedGNN
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For the agent framework (optional):
```bash
cd agents
pip install -r requirements.txt
```

## 7. Data Requirements

The system expects process event logs in CSV format with the following structure:
- case_id: Process instance identifier
- task_name: Activity name
- timestamp: Activity timestamp
- resource: Resource identifier
- amount: Numerical attribute (if applicable)

The AdvancedGNN branch supports multiple benchmark datasets including BPI 2012, BPI 2019, and BPI 2020.

## 8. Usage

### Basic Usage
```bash
python main.py --data_path <input-file-path>
```

### Advanced Usage (AdvancedGNN branch)
```bash
python main.py --data_path <input-file-path> --sample_size 2000 --batch_size 32 \
               --num_epochs 100 --learning_rate 0.001 --hidden_dim 64 \
               --num_layers 2 --heads 4 --dropout 0.5 --use_norm_features
```

### Agent Framework Usage (New)
```bash
cd agents
python src/main.py
```

Results are stored in timestamped directories under `results/` with the following structure:
```
results/run_timestamp/
├── models/          # Trained model weights
├── visualizations/  # Generated visualizations
├── metrics/         # Performance metrics (JSON format)
├── analysis/        # Detailed analysis results
└── experiment_summary.txt  # Comprehensive experiment documentation
```

## 9. Technical Details

### Graph Neural Network Architecture
- Multi-head attention mechanisms
- Dynamic graph construction
- Adaptive feature learning
- Custom loss functions for process-specific metrics
- **New in AdvancedGNN:**
  - Activity group embeddings
  - Layer normalization for training stability
  - Residual connections for better gradient flow
  - GELU activation functions
  - Early stopping with patience

### LSTM Implementation
- Bidirectional sequence modeling
- Variable-length sequence handling
- Custom embedding layer for process activities
- **New in AdvancedGNN:**
  - Improved sequence padding and handling
  - Enhanced evaluation procedures

### Process Mining Components
- Inductive miner implementation
- Token-based replay
- Custom conformance checking metrics
- Advanced bottleneck detection algorithms
- **New in AdvancedGNN:**
  - Semantic activity grouping
  - More detailed cluster analysis
  - Comprehensive bottleneck detection

### Reinforcement Learning
- Custom environment for process optimization
- State-action space modeling
- Policy gradient methods
- Resource allocation optimization

### Agent Framework (New in AdvancedGNN)
- **Autonomous Agents**: Intelligent agents that can monitor and act on process insights
- **Memory System**: Persistent storage for agent knowledge and experiences
- **Process Integration**: Direct integration with GNN process mining models
- **API Endpoints**: RESTful interfaces for interacting with agents and processes
- **Vector Database**: Semantic search capabilities for process knowledge

## 10. Activity Groups (New in AdvancedGNN)

Activity groups provide semantic categorization of business process activities, enabling the model to better understand the functional relationships between tasks. The groups are defined in `activity_groups.py` and include categories such as:

- Order creation activities
- Change management activities
- Approval and cancellation activities
- Receipt recording activities
- Vendor interactions
- System interactions

These groups are automatically integrated into the model architecture via group embeddings and group-aware attention mechanisms.

## 11. Command Line Arguments (New in AdvancedGNN)

The enhanced version provides flexible experimentation through command line arguments:

```bash
python main.py --help
```

Key arguments include:
- `--data_path`: Path to input event log
- `--sample_size`: Number of cases to sample
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--hidden_dim`: Hidden dimension size
- `--num_layers`: Number of GNN layers
- `--heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--use_norm_features`: Use L2 normalized features
- `--additional_features`: Additional numerical features to include

## 12. Contributing

We welcome contributions from the research community. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request with detailed documentation

## 13. Citation

If you use this code in your research, please cite:

```bibtex
@software{GNN_ProcessMining,
  author = {Shashank Dixit/Somesh Misra},
  title = {Process Mining with Graph Neural Networks},
  year = {2025},
  publisher = {ERP.AI},
  url = {https://github.com/ERPdotAI/GNN}
}
``` 
