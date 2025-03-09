import pandas as pd
import numpy as np
import json
from pm4py.objects.log.importer.xes import importer as xes_importer
import pm4py
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer
import torch
from torch_geometric.data import Data
from ..activity_groups import get_activity_group, get_group_id, get_num_groups

def load_and_preprocess_data(data_path, column_mapping=None, required_cols=None, sample_size=None):
    """
    Load and preprocess event log data from various formats (CSV, XES)
    
    Parameters:
    -----------
    data_path : str
        Path to the event log file (CSV or XES)
    column_mapping : str or dict
        Path to JSON file containing column mappings or the mapping dictionary itself
    required_cols : list
        List of required columns (uses default if None)
    sample_size : int
        Number of cases to sample (None for all data)
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed event log
    """
    print(f"Loading data from {data_path}...")
    
    if required_cols is None:
        required_cols = ["case_id", "task_name", "timestamp"]
        
    # Default column mapping for core columns
    default_mapping = {
        "case_id": "case:concept:name",
        "task_name": "concept:name",
        "timestamp": "time:timestamp",
        "resource": "org:resource",
        "amount": "case:Amount"
    }
    
    # Load column mapping from JSON if it's a string path
    if isinstance(column_mapping, str):
        print(f"Loading column mapping from {column_mapping}")
        try:
            with open(column_mapping, 'r') as f:
                column_mapping = json.load(f)
            print(f"Loaded column mapping: {column_mapping}")
        except Exception as e:
            print(f"Error loading column mapping: {e}")
            column_mapping = default_mapping
    
    # Use default mapping if none provided
    if column_mapping is None:
        column_mapping = default_mapping.copy()
    
    # Extract core columns (excluding feature lists)
    core_columns = ["case_id", "task_name", "timestamp", "resource", "amount"]
    core_mappings = {}
    for col in core_columns:
        if col in column_mapping and not isinstance(column_mapping[col], list):
            core_mappings[col] = column_mapping[col]
        elif col in default_mapping:
            core_mappings[col] = default_mapping[col]
    
    print(f"Core mappings: {core_mappings}")
    
    # Load data based on file extension
    file_ext = data_path.split('.')[-1].lower()
    
    if file_ext == 'csv':
        df = pd.read_csv(data_path)
    elif file_ext in ['xes', 'xml']:
        print("Importing XES file...")
        log = xes_importer.apply(data_path)
        print("Converting to dataframe...")
        df = pm4py.convert_to_dataframe(log)
        print(f"Initial dataframe shape: {df.shape}")
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    print(f"Initial columns: {df.columns.tolist()}")
    
    # Sample cases if requested
    if sample_size is not None:
        print(f"Sampling {sample_size} cases...")
        case_col = core_mappings["case_id"]
        case_ids = df[case_col].unique()
        if len(case_ids) > sample_size:
            sampled_cases = np.random.choice(case_ids, size=sample_size, replace=False)
            df = df[df[case_col].isin(sampled_cases)]
    
    print("Renaming columns...")
    # Create reverse mapping only for core columns
    reverse_mapping = {v: k for k, v in core_mappings.items()}
    print(f"Reverse mapping: {reverse_mapping}")
    df.rename(columns=reverse_mapping, inplace=True, errors="ignore")
    
    print(f"Columns after renaming: {df.columns.tolist()}")
    
    # Validate required columns
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
    
    print("Processing timestamps...")
    # Process timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values(["case_id", "timestamp"], inplace=True)
    
    # Handle optional columns
    if "resource" not in df.columns:
        df["resource"] = "UNKNOWN"
    if "amount" not in df.columns:
        df["amount"] = 0.0
    
    print("Processing additional features...")
    # Convert additional features to numeric where possible
    if "additional_features" in column_mapping:
        for feat in column_mapping["additional_features"]:
            if feat in df.columns:
                try:
                    df[feat] = pd.to_numeric(df[feat], errors='coerce')
                    print(f"Converted {feat} to numeric")
                except:
                    print(f"Warning: Could not convert {feat} to numeric")
    
    if "offer_features" in column_mapping:
        for feat in column_mapping["offer_features"]:
            if feat in df.columns:
                try:
                    df[feat] = pd.to_numeric(df[feat], errors='coerce')
                    print(f"Converted {feat} to numeric")
                except:
                    print(f"Warning: Could not convert {feat} to numeric")
    
    print(f"Final dataframe shape: {df.shape}")
    return df 

def validate_event_log(df):
    """
    Validate event log for basic process mining requirements
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input event log
        
    Returns:
    --------
    dict
        Dictionary containing validation statistics and warnings
    """
    stats = {
        "num_cases": int(df["case_id"].nunique()),
        "num_events": int(len(df)),
        "num_activities": int(df["task_name"].nunique()),
        "date_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "validation_warnings": []
    }
    
    # Check for case variants
    case_variants = df.groupby("case_id")["task_name"].agg(list)
    stats["num_variants"] = int(len(set(map(tuple, case_variants))))
    
    # Check for potential issues
    if stats["num_cases"] < 10:
        stats["validation_warnings"].append("Very few cases (<10)")
    if stats["num_activities"] < 3:
        stats["validation_warnings"].append("Very few activities (<3)")
    
    # Check for timestamp issues
    time_diffs = df.groupby("case_id")["timestamp"].diff()
    if (time_diffs < pd.Timedelta(0)).any():
        stats["validation_warnings"].append("Found negative time differences between events")
    
    return stats

def create_feature_representation(df, use_norm_features=True, feature_cols=None):
    """
    Create scaled or normalized feature representation with activity groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input event log
    use_norm_features : bool
        Whether to use L2 normalization (True) or MinMax scaling (False)
    feature_cols : list
        List of additional numerical columns to include in features
        
    Returns:
    --------
    pd.DataFrame, LabelEncoder, LabelEncoder
        Processed dataframe, task encoder, resource encoder
    """
    # Time features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_of_day"] = df["timestamp"].dt.hour
    
    # Encode categorical columns
    le_task = LabelEncoder()
    le_resource = LabelEncoder()
    
    df["task_id"] = le_task.fit_transform(df["task_name"])
    df["resource_id"] = le_resource.fit_transform(df["resource"])
    
    # Add activity group encoding
    df["activity_group"] = df["task_name"].apply(get_activity_group)
    df["group_id"] = df["activity_group"].apply(get_group_id)
    
    # Normalize group IDs to [0,1]
    num_groups = get_num_groups()
    df["group_id_norm"] = df["group_id"] / (num_groups - 1)
    
    # Next task
    df["next_task"] = df.groupby("case_id")["task_id"].shift(-1)
    df.dropna(subset=["next_task"], inplace=True)
    df["next_task"] = df["next_task"].astype(int)
    
    # Base feature columns - now including group ID
    base_features = ["task_id", "group_id_norm", "resource_id", "day_of_week", "hour_of_day"]
    
    # Add amount if present and any additional numerical features
    if "amount" in df.columns:
        base_features.append("amount")
    if feature_cols:
        for col in feature_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                base_features.append(col)
    
    raw_features = df[base_features].values
    
    # Feature scaling/normalization
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(raw_features)
    
    if use_norm_features:
        normalizer = Normalizer(norm='l2')
        features_final = normalizer.fit_transform(raw_features)
    else:
        features_final = features_scaled
    
    # Add features back to dataframe
    for i, col in enumerate(base_features):
        df[f"feat_{col}"] = features_final[:,i]
    
    # Add group information to metadata
    df.attrs['num_groups'] = num_groups
    df.attrs['group_mapping'] = {group: idx for idx, group in enumerate(df['activity_group'].unique())}
    
    return df, le_task, le_resource

def build_graph_data(df):
    """
    Convert preprocessed data into graph format for GNN with activity groups
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed event log with feature columns and group information
        
    Returns:
    --------
    list
        List of torch_geometric.data.Data objects representing process graphs
    """
    graphs = []
    for cid, cdata in df.groupby("case_id"):
        cdata.sort_values("timestamp", inplace=True)

        # Get node features including group information
        x_data = torch.tensor(cdata[[
            "feat_task_id",
            "feat_group_id_norm",
            "feat_resource_id",
            "feat_day_of_week",
            "feat_hour_of_day"
        ]].values, dtype=torch.float)

        # Get group IDs for each node
        group_ids = torch.tensor(cdata["group_id"].values, dtype=torch.long)

        n_nodes = len(cdata)
        if n_nodes > 1:
            # Create edges for sequential flow
            src = list(range(n_nodes-1))
            tgt = list(range(1,n_nodes))
            edge_index = torch.tensor([src+tgt, tgt+src], dtype=torch.long)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
            
        y_data = torch.tensor(cdata["next_task"].values, dtype=torch.long)
        
        # Create graph with group IDs
        data_obj = Data(
            x=x_data,
            edge_index=edge_index,
            y=y_data,
            group_ids=group_ids
        )
        graphs.append(data_obj)

    return graphs

def compute_class_weights(df, num_classes):
    """
    Compute balanced class weights for training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed event log with next_task column
    num_classes : int
        Total number of task classes
        
    Returns:
    --------
    torch.Tensor
        Class weights tensor for balanced training
    """
    from sklearn.utils.class_weight import compute_class_weight
    train_labels = df["next_task"].values
    class_weights = np.ones(num_classes, dtype=np.float32)
    present = np.unique(train_labels)
    cw = compute_class_weight("balanced", classes=present, y=train_labels)
    for i, cval in enumerate(present):
        class_weights[cval] = cw[i]
    return torch.tensor(class_weights, dtype=torch.float32) 