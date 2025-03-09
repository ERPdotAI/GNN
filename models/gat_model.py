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

class NextTaskGAT(nn.Module):
    """
    Graph Attention Network for next task prediction with activity groups
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_groups, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        
        # Activity group embedding
        self.group_embedding = nn.Embedding(num_groups, hidden_dim)
        
        # Input projection with group attention
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # GAT layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First GAT layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=True))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))
        
        # Additional GAT layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.best_val_loss = float('inf')

    def forward(self, x, edge_index, batch, group_ids=None):
        # Get group embeddings if provided
        if group_ids is not None:
            group_emb = self.group_embedding(group_ids)
            x = torch.cat([x, group_emb], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        
        # GAT layers with residual connections and layer normalization
        for conv, norm in zip(self.convs, self.norms):
            identity = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            if x.size(-1) == identity.size(-1):  # Only add residual if dimensions match
                x = x + identity
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output projection
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def train_gat_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, model_path=None):
    """
    Train the GAT model with activity groups
    """
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass with group IDs if available
            if hasattr(batch, 'group_ids'):
                out = model(batch.x, batch.edge_index, batch.batch, batch.group_ids)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            
            loss = criterion(out, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if hasattr(batch, 'group_ids'):
                    out = model(batch.x, batch.edge_index, batch.batch, batch.group_ids)
                else:
                    out = model(batch.x, batch.edge_index, batch.batch)
                val_loss += criterion(out, batch.y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.best_val_loss = best_val_loss
            patience_counter = 0
            if model_path:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Load best model if saved
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    return model

def evaluate_gat_model(model, loader, device):
    """
    Evaluate the GAT model
    """
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if hasattr(batch, 'group_ids'):
                out = model(batch.x, batch.edge_index, batch.batch, batch.group_ids)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            
            prob = torch.exp(out)
            pred = prob.argmax(dim=1)
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_prob) 