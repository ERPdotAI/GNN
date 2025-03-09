#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph Attention Network (GAT) model for process mining with activity groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np
from visualization.process_viz import plot_detailed_confusion_matrix

class NextTaskGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # GAT layers with multiple heads
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=0.2)
        
        # Layer normalization after each GAT layer
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights carefully
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, edge_index, batch=None):
        # Input normalization
        x = self.input_norm(x)
        
        # First GAT layer
        x1 = self.gat1(x, edge_index)
        x1 = self.norm1(x1)
        x1 = F.gelu(x1)
        
        # Second GAT layer
        x2 = self.gat2(x1, edge_index)
        x2 = self.norm2(x2)
        x2 = F.gelu(x2)
        
        # Output projection
        out = self.output(x2)
        
        return F.log_softmax(out, dim=1)

def train_gat_model(model, train_loader, test_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on device: {device}")
    
    # Optimizer with smaller learning rate and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    training_losses = []  # Track epoch losses
    
    def diagnose_nan(tensor, name=""):
        if torch.isnan(tensor).any():
            print(f"NaN in {name}: Shape={tensor.shape}, Range=[{tensor.min()}, {tensor.max()}]")
            return True
        return False
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, data in enumerate(train_loader):
            try:
                data = data.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                out = model(data.x, data.edge_index, data.batch)
                
                # Check for NaN values
                if diagnose_nan(out, "model output"):
                    continue
                
                # Loss calculation
                loss = F.nll_loss(out, data.y)
                
                if diagnose_nan(loss, "loss"):
                    continue
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                if valid_batches % 10 == 0:
                    print(f"Batch {valid_batches}, Loss: {loss.item():.4f}")
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            training_losses.append(avg_loss)  # Store the average loss
            print(f'Epoch {epoch+1:03d}, Loss: {avg_loss:.4f}, Valid batches: {valid_batches}')
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                print(f"New best loss: {best_loss:.4f}")
                # Save best model
                torch.save(model.state_dict(), 'best_gat_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        else:
            print(f'Epoch {epoch+1:03d}, No valid batches')
    
    return model, training_losses

def evaluate_model(model, data_loader, device, class_mapping):
    """Evaluate model and generate confusion matrix"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            # Properly pass x, edge_index, and batch to the model
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    
    # Get activity names from task IDs
    activity_names = sorted(class_mapping.keys())
    
    # Generate and save confusion matrix
    plot_detailed_confusion_matrix(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=activity_names,
        save_path="results/plots/confusion_matrix.png"
    )
    
    # Calculate accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    
    return accuracy, all_preds, all_labels 