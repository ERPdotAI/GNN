#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script for enhanced process mining with GNN, LSTM, and RL
"""

import os
import torch
import random
import numpy as np
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from datetime import datetime
import json
import shutil
from sklearn.metrics import mean_absolute_error, r2_score, log_loss, accuracy_score, matthews_corrcoef
import torch.nn.functional as F

# Import local modules
from modules.data_preprocessing import (
    load_and_preprocess_data,
    create_feature_representation,
    build_graph_data,
    compute_class_weights,
    validate_event_log
)
from models.gat_model import (
    NextTaskGAT,
    train_gat_model,
    evaluate_gat_model
)
from models.lstm_model import (
    NextActivityLSTM,
    prepare_sequence_data,
    make_padded_dataset,
    train_lstm_model,
    evaluate_lstm_model
)
from modules.process_mining import (
    analyze_bottlenecks,
    analyze_cycle_times,
    analyze_rare_transitions,
    perform_conformance_checking,
    analyze_transition_patterns,
    spectral_cluster_graph,
    build_task_adjacency,
    analyze_cluster_statistics,
    print_cluster_analysis
)
from visualization.process_viz import (
    plot_confusion_matrix,
    plot_embeddings,
    plot_cycle_time_distribution,
    plot_process_flow,
    plot_transition_heatmap,
    create_sankey_diagram
)
from activity_groups import get_num_groups

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Mining with GNN')
    parser.add_argument('--data_path', type=str, default='BPI_Challenge_2019.xes',
                      help='Path to input event log')
    parser.add_argument('--sample_size', type=int, default=2000,
                      help='Number of cases to sample')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of GNN layers')
    parser.add_argument('--heads', type=int, default=4,
                      help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    parser.add_argument('--use_norm_features', action='store_true',
                      help='Use L2 normalized features')
    parser.add_argument('--additional_features', nargs='+',
                      help='Additional numerical features to include')
    return parser.parse_args()

def save_metrics(metrics, run_dir, filename):
    """Save metrics to JSON file"""
    metrics_path = os.path.join(run_dir, "metrics", filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"run_{timestamp}")
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_data(args.data_path)
    
    # Validate event log
    validate_event_log(df)
    
    # Create feature representation with activity groups
    df, le_task, le_resource = create_feature_representation(
        df,
        use_norm_features=args.use_norm_features,
        feature_cols=args.additional_features
    )
    
    # Save preprocessing info
    preproc_info = {
        "num_tasks": len(le_task.classes_),
        "num_resources": len(le_resource.classes_),
        "num_groups": df.attrs['num_groups'],
        "group_mapping": df.attrs['group_mapping'],
        "num_cases": df["case_id"].nunique(),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "scaling": "L2 Normalization" if args.use_norm_features else "MinMax",
        "additional_features": args.additional_features
    }
    with open(os.path.join(run_dir, "metrics", "preprocessing_info.json"), 'w') as f:
        json.dump(preproc_info, f, indent=4)
    
    # 2. Build graph data
    print("\n2. Building graph data...")
    graphs = build_graph_data(df)
    train_size = int(len(graphs)*0.8)
    train_graphs = graphs[:train_size]
    val_graphs = graphs[train_size:]
    
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    
    # 3. Train GAT model
    print("\n3. Training GAT model...")
    num_classes = len(le_task.classes_)
    num_groups = get_num_groups()
    class_weights = compute_class_weights(df, num_classes).to(device)
    
    # Initialize model with group information
    gat_model = NextTaskGAT(
        input_dim=5,  # task_id, resource_id, day_of_week, hour_of_day, amount
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_groups=num_groups,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(gat_model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    
    gat_model_path = os.path.join(run_dir, "models", "best_gnn_model.pth")
    gat_model = train_gat_model(
        gat_model, train_loader, val_loader,
        criterion, optimizer, device,
        num_epochs=args.num_epochs,
        model_path=gat_model_path
    )
    
    # 4. Evaluate GAT model
    print("\n=== GAT Next-Activity Prediction ===")
    y_true, y_pred, y_prob = evaluate_gat_model(gat_model, val_loader, device)
    
    # Calculate metrics
    gat_accuracy = float(accuracy_score(y_true, y_pred))
    gat_mcc = float(matthews_corrcoef(y_true, y_pred))
    
    print(f"Number of classes: {num_classes}")
    print(f"Accuracy: {gat_accuracy:.4f}")
    print(f"MCC: {gat_mcc:.4f}")
    
    # Save GAT metrics
    gat_metrics = {
        "accuracy": gat_accuracy,
        "mcc": gat_mcc,
        "num_classes": num_classes,
        "num_groups": num_groups,
        "best_val_loss": float(gat_model.best_val_loss)
    }
    save_metrics(gat_metrics, run_dir, "gat_metrics.json")
    
    # 5. Train LSTM model
    print("\n4. Training LSTM model...")
    train_seq, test_seq = prepare_sequence_data(df)
    X_train_pad, X_train_len, y_train, max_len = make_padded_dataset(train_seq, num_classes)
    X_test_pad, X_test_len, y_test, _ = make_padded_dataset(test_seq, num_classes)
    
    lstm_model = NextActivityLSTM(
        num_classes,
        emb_dim=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)
    
    lstm_model_path = os.path.join(run_dir, "models", "lstm_next_activity.pth")
    lstm_model = train_lstm_model(
        lstm_model, 
        X_train_pad.to(device), 
        X_train_len.to(device), 
        y_train.to(device),
        device,
        batch_size=args.batch_size,
        epochs=args.num_epochs,
        model_path=lstm_model_path
    )
    
    # 6. Evaluate LSTM model
    print("\n=== LSTM Next-Activity Prediction ===")
    lstm_preds, lstm_probs = evaluate_lstm_model(
        lstm_model, 
        X_test_pad.to(device),
        X_test_len.to(device),
        args.batch_size,
        device
    )
    
    lstm_accuracy = float(accuracy_score(y_test.numpy(), lstm_preds))
    lstm_mcc = float(matthews_corrcoef(y_test.numpy(), lstm_preds))
    
    print(f"Accuracy: {lstm_accuracy:.4f}")
    print(f"MCC: {lstm_mcc:.4f}")
    
    # Save LSTM metrics
    lstm_metrics = {
        "accuracy": lstm_accuracy,
        "mcc": lstm_mcc,
        "num_classes": num_classes,
        "num_groups": num_groups
    }
    save_metrics(lstm_metrics, run_dir, "lstm_metrics.json")
    
    # 7. Process Mining Analysis
    print("\n5. Performing process mining analysis...")
    bottleneck_stats, significant_bottlenecks = analyze_bottlenecks(df)
    case_merged, long_cases, cut95 = analyze_cycle_times(df)
    
    # Save detailed results to a summary file
    with open(os.path.join(run_dir, "experiment_summary.txt"), "w") as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"Scaling: {'L2 Normalization' if args.use_norm_features else 'MinMax scaling'}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Number of activity groups: {num_groups}\n\n")
        
        f.write("=== GAT Next-Activity Prediction ===\n")
        f.write(f"Accuracy: {gat_accuracy:.4f}\n")
        f.write(f"MCC: {gat_mcc:.4f}\n")
        f.write(f"Best validation loss: {gat_model.best_val_loss:.4f}\n\n")
        
        f.write("=== LSTM Next-Activity Prediction ===\n")
        f.write(f"Accuracy: {lstm_accuracy:.4f}\n")
        f.write(f"MCC: {lstm_mcc:.4f}\n\n")
        
        f.write("=== Activity Groups ===\n")
        for group, idx in df.attrs['group_mapping'].items():
            f.write(f"{group}: {idx}\n")
    
    # 8. Visualizations
    print("\n6. Creating visualizations...")
    viz_dir = os.path.join(run_dir, "visualizations")
    plot_cycle_time_distribution(
        case_merged["duration_h"].values,
        os.path.join(viz_dir, "cycle_time_distribution.png")
    )
    plot_process_flow(
        bottleneck_stats, le_task, significant_bottlenecks.head(),
        os.path.join(viz_dir, "process_flow_bottlenecks.png")
    )
    
    # Get transition patterns
    transitions, trans_count, prob_matrix = analyze_transition_patterns(df)
    plot_transition_heatmap(
        transitions, le_task,
        os.path.join(viz_dir, "transition_probability_heatmap.png")
    )
    create_sankey_diagram(
        transitions, le_task,
        os.path.join(viz_dir, "process_flow_sankey.html")
    )
    
    print(f"\nDone! Results saved in {run_dir}")

if __name__ == "__main__":
    main() 