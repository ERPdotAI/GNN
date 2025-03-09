#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Mining Analysis Module
Includes bottleneck analysis, conformance checking, and cycle time analysis
"""

import pandas as pd
import numpy as np
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

def analyze_bottlenecks(df, freq_threshold=5):
    """
    Analyze process bottlenecks based on waiting times between activities
    """
    df = df.copy()
    df["next_task_id"] = df.groupby("case_id")["task_id"].shift(-1)
    df["next_timestamp"] = df.groupby("case_id")["timestamp"].shift(-1)
    transitions = df.dropna(subset=["next_task_id"]).copy()
    transitions["wait_sec"] = (transitions["next_timestamp"] - transitions["timestamp"]).dt.total_seconds()
    
    bottleneck_stats = transitions.groupby(["task_id","next_task_id"])["wait_sec"].agg([
        "mean","count"
    ]).reset_index()
    
    bottleneck_stats["mean_hours"] = bottleneck_stats["mean"]/3600.0
    bottleneck_stats.sort_values("mean_hours", ascending=False, inplace=True)
    
    # Filter by frequency threshold
    significant_bottlenecks = bottleneck_stats[bottleneck_stats["count"] >= freq_threshold]
    
    return bottleneck_stats, significant_bottlenecks

def analyze_cycle_times(df):
    """
    Analyze process cycle times
    """
    case_grouped = df.groupby("case_id")["timestamp"].agg(["min","max"])
    case_grouped["cycle_time_hours"] = (
        case_grouped["max"] - case_grouped["min"]
    ).dt.total_seconds()/3600.0
    case_grouped.reset_index(inplace=True)
    
    df_feats = df.groupby("case_id").agg({
        "amount": "mean",
        "task_id": "count"
    }).rename(columns={
        "amount": "mean_amount",
        "task_id": "num_events"
    }).reset_index()
    
    case_merged = pd.merge(case_grouped, df_feats, on="case_id", how="left")
    case_merged["duration_h"] = case_merged["cycle_time_hours"]
    
    # Identify long-running cases (95th percentile)
    cut95 = case_merged["duration_h"].quantile(0.95)
    long_cases = case_merged[case_merged["duration_h"] > cut95]
    
    return case_merged, long_cases, cut95

def analyze_rare_transitions(bottleneck_stats, rare_threshold=2):
    """
    Identify rare transitions in the process
    """
    rare_trans = bottleneck_stats[bottleneck_stats["count"] <= rare_threshold]
    return rare_trans

def perform_conformance_checking(df):
    """
    Perform conformance checking using inductive miner and token replay
    """
    df_pm = df[["case_id","task_name","timestamp"]].rename(columns={
        "case_id": "case:concept:name",
        "task_name": "concept:name",
        "timestamp": "time:timestamp"
    })
    
    df_pm = dataframe_utils.convert_timestamp_columns_in_df(df_pm)
    event_log = log_converter.apply(df_pm)
    
    process_tree = inductive_miner.apply(event_log)
    from pm4py.objects.conversion.process_tree import converter as pt_converter
    net, im, fm = pt_converter.apply(process_tree)
    
    replayed = token_replay.apply(event_log, net, im, fm)
    n_deviant = sum(1 for t in replayed if not t["trace_is_fit"])
    
    return replayed, n_deviant

def analyze_transition_patterns(df):
    """
    Analyze transition patterns and compute transition matrix
    """
    transitions = df.copy()
    transitions["next_task_id"] = transitions.groupby("case_id")["task_id"].shift(-1)
    trans_count = transitions.groupby(["task_id","next_task_id"]).size().unstack(fill_value=0)
    prob_matrix = trans_count.div(trans_count.sum(axis=1), axis=0)
    
    return transitions, trans_count, prob_matrix

def spectral_cluster_graph(adj_matrix, k=2):
    """
    Perform spectral clustering on process graph
    """
    from sklearn.cluster import KMeans
    
    degrees = np.sum(adj_matrix, axis=1)
    D = np.diag(degrees)
    L = D - adj_matrix  # unnormalized Laplacian

    eigenvals, eigenvecs = np.linalg.eig(L)
    idx = np.argsort(eigenvals)
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]

    if k == 2:
        # Fiedler vector = second smallest eigenvector
        fiedler_vec = np.real(eigenvecs[:, 1])
        # Partition by sign
        labels = (fiedler_vec >= 0).astype(int)
    else:
        # multi-cluster
        embedding = np.real(eigenvecs[:, 1:k])
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(embedding)
        labels = kmeans.labels_
        
    return labels

def build_task_adjacency(df, num_tasks):
    """
    Build adjacency matrix weighted by transition frequencies
    """
    A = np.zeros((num_tasks, num_tasks), dtype=np.float32)
    for cid, cdata in df.groupby("case_id"):
        cdata = cdata.sort_values("timestamp")
        tasks_seq = cdata["task_id"].values
        for i in range(len(tasks_seq)-1):
            src = tasks_seq[i]
            tgt = tasks_seq[i+1]
            A[src, tgt] += 1.0
    return A 

def analyze_cluster_statistics(df, cluster_labels, le_task):
    """
    Analyze statistics for each cluster
    """
    # Add cluster labels to dataframe
    df = df.copy()
    df["cluster"] = df["task_id"].apply(lambda x: cluster_labels[x])
    
    # Cluster sizes and tasks
    cluster_stats = {}
    for cluster_id in sorted(set(cluster_labels)):
        task_ids = [i for i, lbl in enumerate(cluster_labels) if lbl == cluster_id]
        task_names = [le_task.inverse_transform([tid])[0] for tid in task_ids]
        
        # Get events and cases in this cluster
        cluster_events = df[df["cluster"] == cluster_id]
        cluster_cases = cluster_events["case_id"].nunique()
        
        # Calculate waiting times
        cluster_events["next_timestamp"] = cluster_events.groupby("case_id")["timestamp"].shift(-1)
        cluster_events["wait_hours"] = (cluster_events["next_timestamp"] - cluster_events["timestamp"]).dt.total_seconds() / 3600
        avg_wait = cluster_events["wait_hours"].mean()
        
        cluster_stats[cluster_id] = {
            "size": len(task_ids),
            "tasks": task_names,
            "num_events": len(cluster_events),
            "num_cases": cluster_cases,
            "avg_wait_hours": avg_wait if not pd.isna(avg_wait) else 0.0
        }
    
    return cluster_stats

def print_cluster_analysis(cluster_stats):
    """Print detailed cluster analysis"""
    print("\n=== Cluster Analysis ===")
    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {stats['size']} tasks")
        print(f"  Events: {stats['num_events']}")
        print(f"  Cases: {stats['num_cases']}")
        print(f"  Avg Wait Time: {stats['avg_wait_hours']:.2f} hours")
        print("  Tasks:")
        for task in stats['tasks']:
            print(f"    - {task}") 