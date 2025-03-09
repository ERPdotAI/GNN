#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector database for storing and retrieving process embeddings.
This module handles the storage and retrieval of embeddings for process designs.
"""

import os
import json
import logging
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using in-memory vector store only.")

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for process embeddings using FAISS."""
    
    def __init__(self, 
                 dimension: int = 512, 
                 index_type: str = "Flat", 
                 storage_dir: Optional[str] = None):
        """
        Initialize a vector store.
        
        Args:
            dimension: Dimensionality of the embeddings
            index_type: Type of FAISS index (Flat, IVF, HNSW)
            storage_dir: Directory to store the index and metadata
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required for VectorStore. Install with 'pip install faiss-cpu'")
        
        self.dimension = dimension
        self.index_type = index_type
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            
        # Initialize the index
        self.index = None
        self._init_index()
        
        # Mapping from process IDs to index positions
        self.id_to_index = {}
        
        # Mapping from index positions to process IDs
        self.index_to_id = []
        
        # Metadata storage
        self.metadata = {}
        
        # Load existing data if available
        if storage_dir:
            index_path = os.path.join(storage_dir, "index.faiss")
            metadata_path = os.path.join(storage_dir, "metadata.json")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load index
                self.index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    self.metadata = data["metadata"]
                    self.id_to_index = data["id_to_index"]
                    self.index_to_id = data["index_to_id"]
    
    def _init_index(self):
        """Initialize the FAISS index."""
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # IVF index requires training, so we start with a flat index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.nprobe = 10  # Number of clusters to visit during search
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            # Default to flat index
            logger.warning(f"Unknown index type {self.index_type}, using Flat index")
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def save(self):
        """Save the index and metadata to disk."""
        if not self.storage_dir:
            logger.warning("No storage directory specified, cannot save")
            return False
        
        try:
            # Save index
            index_path = os.path.join(self.storage_dir, "index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.storage_dir, "metadata.json")
            data = {
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id
            }
            with open(metadata_path, "w") as f:
                json.dump(data, f)
            
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def add(self, 
            embedding: np.ndarray, 
            metadata: Dict[str, Any], 
            process_id: Optional[str] = None) -> str:
        """
        Add a process embedding to the store.
        
        Args:
            embedding: The process embedding
            metadata: Metadata about the process
            process_id: Optional process ID (will be generated if not provided)
            
        Returns:
            The process ID
        """
        # Generate process ID if not provided
        if process_id is None:
            process_id = str(uuid.uuid4())
        
        # Check if process ID already exists
        if process_id in self.id_to_index:
            logger.warning(f"Process ID {process_id} already exists, updating")
            return self.update(process_id, embedding, metadata)
        
        # Make sure embedding is the right shape
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.shape != (self.dimension,):
            # Reshape if a batch of 1
            if embedding.shape == (1, self.dimension):
                embedding = embedding.reshape(self.dimension)
            else:
                raise ValueError(f"Embedding has wrong shape: {embedding.shape}, expected ({self.dimension},)")
        
        # Add to index
        embedding = embedding.astype(np.float32).reshape(1, -1)  # Reshape for faiss
        self.index.add(embedding)
        
        # Update mappings
        index_position = self.index.ntotal - 1
        self.id_to_index[process_id] = index_position
        self.index_to_id.append(process_id)
        
        # Store metadata
        self.metadata[process_id] = metadata
        
        return process_id
    
    def update(self, 
               process_id: str, 
               embedding: np.ndarray, 
               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a process embedding.
        
        Args:
            process_id: ID of the process to update
            embedding: The new embedding
            metadata: Optional new metadata (if None, existing metadata is kept)
            
        Returns:
            True if successful, False otherwise
        """
        if process_id not in self.id_to_index:
            logger.warning(f"Process ID {process_id} not found, cannot update")
            return False
        
        # For FAISS, we can't update in place, so we remove and re-add
        # Delete the old embedding
        success = self.delete(process_id)
        if not success:
            return False
        
        # Keep the existing metadata if not provided
        if metadata is None:
            metadata = self.metadata.get(process_id, {})
        
        # Add the new embedding
        self.add(embedding, metadata, process_id)
        
        return True
    
    def get(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a process by ID.
        
        Args:
            process_id: ID of the process to get
            
        Returns:
            Dictionary with metadata, or None if not found
        """
        if process_id not in self.metadata:
            return None
        
        return self.metadata[process_id]
    
    def delete(self, process_id: str) -> bool:
        """
        Delete a process.
        
        Args:
            process_id: ID of the process to delete
            
        Returns:
            True if successful, False otherwise
        """
        if process_id not in self.id_to_index:
            logger.warning(f"Process ID {process_id} not found, cannot delete")
            return False
        
        # FAISS doesn't support deletion, so we have to recreate the index
        # Get all embeddings except the one to delete
        index_position = self.id_to_index[process_id]
        
        # Create a new index
        old_index = self.index
        self._init_index()
        
        # Get all process IDs excluding the one to delete
        new_index_to_id = []
        new_id_to_index = {}
        
        # For each process, if it's not the one to delete, add it to the new index
        for i, pid in enumerate(self.index_to_id):
            if pid != process_id:
                # Get the embedding from the old index
                if i < index_position:
                    # Before the deleted item, index is the same
                    new_index = i
                else:
                    # After the deleted item, index is one less
                    new_index = i - 1
                
                # Add to the new mappings
                new_index_to_id.append(pid)
                new_id_to_index[pid] = new_index
        
        # If there are no items left, reset everything
        if not new_index_to_id:
            self.index_to_id = []
            self.id_to_index = {}
            self.metadata.pop(process_id, None)
            return True
        
        # Add all embeddings except the one to delete
        embeddings = []
        for pid in new_index_to_id:
            idx = self.id_to_index[pid]
            vec = old_index.reconstruct(idx)
            embeddings.append(vec)
        
        # Add all embeddings at once
        embeddings = np.vstack(embeddings).astype(np.float32)
        self.index.add(embeddings)
        
        # Update mappings
        self.index_to_id = new_index_to_id
        self.id_to_index = new_id_to_index
        
        # Remove metadata
        self.metadata.pop(process_id, None)
        
        return True
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5, 
               return_distances: bool = True) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for similar processes.
        
        Args:
            query_embedding: The query embedding
            k: Number of results to return
            return_distances: Whether to return distances
            
        Returns:
            List of process IDs or (process ID, distance) tuples
        """
        if not self.index_to_id:
            logger.warning("Vector store is empty, cannot search")
            return [] if return_distances else []
        
        # Make sure embedding is the right shape
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        if query_embedding.shape != (self.dimension,):
            # Reshape if a batch of 1
            if query_embedding.shape == (1, self.dimension):
                query_embedding = query_embedding.reshape(self.dimension)
            else:
                raise ValueError(f"Embedding has wrong shape: {query_embedding.shape}, expected ({self.dimension},)")
        
        # FAISS needs a 2D array
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Limit k to the number of items in the index
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert indices to process IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.index_to_id):
                continue  # Invalid index
            
            process_id = self.index_to_id[idx]
            if return_distances:
                results.append((process_id, float(distances[0][i])))
            else:
                results.append(process_id)
        
        return results
    
    def search_by_metadata(self, 
                          filter_dict: Dict[str, Any], 
                          k: Optional[int] = None) -> List[str]:
        """
        Search processes by metadata.
        
        Args:
            filter_dict: Dictionary of metadata key-value pairs to match
            k: Maximum number of results to return
            
        Returns:
            List of matching process IDs
        """
        results = []
        
        for process_id, metadata in self.metadata.items():
            # Check if all filter criteria match
            match = True
            for key, value in filter_dict.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                results.append(process_id)
        
        # Limit results if k is specified
        if k is not None:
            results = results[:k]
        
        return results
    
    def get_all_process_ids(self) -> List[str]:
        """
        Get all process IDs in the store.
        
        Returns:
            List of process IDs
        """
        return list(self.metadata.keys())
    
    def count(self) -> int:
        """
        Get the number of processes in the store.
        
        Returns:
            Number of processes
        """
        return len(self.metadata)
    
    def clear(self) -> bool:
        """
        Clear the vector store.
        
        Returns:
            True if successful
        """
        # Create a new index
        self._init_index()
        
        # Reset mappings
        self.id_to_index = {}
        self.index_to_id = []
        self.metadata = {}
        
        # Save if storage is configured
        if self.storage_dir:
            self.save()
        
        return True
    
    def export_metadata(self, file_path: str) -> bool:
        """
        Export metadata to a file.
        
        Args:
            file_path: Path to export to
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exporting metadata: {str(e)}")
            return False
    
    def import_metadata(self, file_path: str) -> bool:
        """
        Import metadata from a file.
        
        Args:
            file_path: Path to import from
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, "r") as f:
                self.metadata = json.load(f)
            
            # Update id_to_index and index_to_id
            # This assumes the embeddings are already in the index
            self.id_to_index = {pid: i for i, pid in enumerate(self.metadata.keys())}
            self.index_to_id = list(self.metadata.keys())
            
            return True
        except Exception as e:
            logger.error(f"Error importing metadata: {str(e)}")
            return False
    
    def __len__(self) -> int:
        """Get the number of processes in the store."""
        return self.count()
    
    def __contains__(self, process_id: str) -> bool:
        """Check if a process ID exists."""
        return process_id in self.metadata

class InMemoryVectorStore:
    """In-memory vector store for process embeddings."""
    
    def __init__(self, dimension: int = 512):
        """
        Initialize an in-memory vector store.
        
        Args:
            dimension: Dimensionality of the embeddings
        """
        self.dimension = dimension
        self.embeddings = {}  # Map from process ID to embedding
        self.metadata = {}    # Map from process ID to metadata
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any], process_id: Optional[str] = None) -> str:
        """
        Add a process embedding to the store.
        
        Args:
            embedding: The process embedding
            metadata: Metadata about the process
            process_id: Optional process ID (will be generated if not provided)
            
        Returns:
            The process ID
        """
        # Generate process ID if not provided
        if process_id is None:
            process_id = str(uuid.uuid4())
        
        # Make sure embedding is the right shape
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        if embedding.shape != (self.dimension,):
            # Reshape if a batch of 1
            if embedding.shape == (1, self.dimension):
                embedding = embedding.reshape(self.dimension)
            else:
                raise ValueError(f"Embedding has wrong shape: {embedding.shape}, expected ({self.dimension},)")
        
        # Store the embedding and metadata
        self.embeddings[process_id] = embedding
        self.metadata[process_id] = metadata
        
        return process_id
    
    def update(self, process_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a process embedding.
        
        Args:
            process_id: ID of the process to update
            embedding: The new embedding
            metadata: Optional new metadata (if None, existing metadata is kept)
            
        Returns:
            True if successful, False otherwise
        """
        if process_id not in self.embeddings:
            logger.warning(f"Process ID {process_id} not found, cannot update")
            return False
        
        # Update embedding
        if embedding is not None:
            # Make sure embedding is the right shape
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            if embedding.shape != (self.dimension,):
                # Reshape if a batch of 1
                if embedding.shape == (1, self.dimension):
                    embedding = embedding.reshape(self.dimension)
                else:
                    raise ValueError(f"Embedding has wrong shape: {embedding.shape}, expected ({self.dimension},)")
            
            self.embeddings[process_id] = embedding
        
        # Update metadata
        if metadata is not None:
            self.metadata[process_id] = metadata
        
        return True
    
    def get(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a process by ID.
        
        Args:
            process_id: ID of the process to get
            
        Returns:
            Dictionary with metadata, or None if not found
        """
        if process_id not in self.metadata:
            return None
        
        return self.metadata[process_id]
    
    def delete(self, process_id: str) -> bool:
        """
        Delete a process.
        
        Args:
            process_id: ID of the process to delete
            
        Returns:
            True if successful, False otherwise
        """
        if process_id not in self.embeddings:
            logger.warning(f"Process ID {process_id} not found, cannot delete")
            return False
        
        # Remove embedding and metadata
        self.embeddings.pop(process_id)
        self.metadata.pop(process_id, None)
        
        return True
    
    def search(self, query_embedding: np.ndarray, k: int = 5, return_distances: bool = True) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for similar processes.
        
        Args:
            query_embedding: The query embedding
            k: Number of results to return
            return_distances: Whether to return distances
            
        Returns:
            List of process IDs or (process ID, distance) tuples
        """
        if not self.embeddings:
            logger.warning("Vector store is empty, cannot search")
            return [] if return_distances else []
        
        # Make sure embedding is the right shape
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        
        if query_embedding.shape != (self.dimension,):
            # Reshape if a batch of 1
            if query_embedding.shape == (1, self.dimension):
                query_embedding = query_embedding.reshape(self.dimension)
            else:
                raise ValueError(f"Embedding has wrong shape: {query_embedding.shape}, expected ({self.dimension},)")
        
        # Compute distances to all embeddings
        distances = {}
        for process_id, embedding in self.embeddings.items():
            distance = np.linalg.norm(embedding - query_embedding)
            distances[process_id] = distance
        
        # Sort by distance
        sorted_items = sorted(distances.items(), key=lambda x: x[1])
        
        # Limit to k results
        sorted_items = sorted_items[:k]
        
        # Format results
        if return_distances:
            return [(process_id, float(distance)) for process_id, distance in sorted_items]
        else:
            return [process_id for process_id, _ in sorted_items]
    
    def search_by_metadata(self, filter_dict: Dict[str, Any], k: Optional[int] = None) -> List[str]:
        """
        Search processes by metadata.
        
        Args:
            filter_dict: Dictionary of metadata key-value pairs to match
            k: Maximum number of results to return
            
        Returns:
            List of matching process IDs
        """
        results = []
        
        for process_id, metadata in self.metadata.items():
            # Check if all filter criteria match
            match = True
            for key, value in filter_dict.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            
            if match:
                results.append(process_id)
        
        # Limit results if k is specified
        if k is not None:
            results = results[:k]
        
        return results
    
    def get_all_process_ids(self) -> List[str]:
        """
        Get all process IDs in the store.
        
        Returns:
            List of process IDs
        """
        return list(self.embeddings.keys())
    
    def count(self) -> int:
        """
        Get the number of processes in the store.
        
        Returns:
            Number of processes
        """
        return len(self.embeddings)
    
    def clear(self) -> bool:
        """
        Clear the vector store.
        
        Returns:
            True if successful
        """
        self.embeddings = {}
        self.metadata = {}
        return True
    
    def __len__(self) -> int:
        """Get the number of processes in the store."""
        return self.count()
    
    def __contains__(self, process_id: str) -> bool:
        """Check if a process ID exists."""
        return process_id in self.embeddings 