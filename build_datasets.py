import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_pyg_data():
    print("--- Loading Graph Data ---")
    try:
        nodes_df = pd.read_csv('tracking_data.csv')
        edges_df = pd.read_csv('graph_edges.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure 'tracking_data.csv' and 'graph_edges.csv' are in the directory.")
        return

    print("--- Building Neural Tensors ---")
    
    # 1. Node Features (x): Tensor shape [num_nodes, 4]
    # We include x, y, z_slice, and time. 
    # We standardize (normalize) them so the Neural Network converges faster.
    x_features = nodes_df[['x', 'y', 'z_slice', 'time']].values
    x_features = (x_features - x_features.mean(axis=0)) / (x_features.std(axis=0) + 1e-8)
    x_tensor = torch.tensor(x_features, dtype=torch.float)
    
    # 2. Edge Index (edge_index): Tensor shape [2, num_edges]
    # This tells PyTorch exactly which nodes are connected.
    source_nodes = edges_df['source_node'].values
    target_nodes = edges_df['target_node'].values
    edge_index = torch.tensor(np.vstack((source_nodes, target_nodes)), dtype=torch.long)
    
    # 3. Edge Attributes (edge_attr): Tensor shape [num_edges, 4]
    # The spatial dynamics: distance, delta_x, delta_y, delta_z.
    edge_features = edges_df[['distance', 'delta_x', 'delta_y', 'delta_z']].values
    edge_features = (edge_features - edge_features.mean(axis=0)) / (edge_features.std(axis=0) + 1e-8)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # 4. Construct the Graph Data Object
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr)
    
    print("--- PyTorch Dataset Summary ---")
    print(f"Total Graph Nodes (Cells): {data.num_nodes}")
    print(f"Total Graph Edges (Connections): {data.num_edges}")
    print(f"Node Feature Tensor Shape: {data.x.shape}")
    print(f"Edge Attribute Tensor Shape: {data.edge_attr.shape}")
    
    # Save the PyTorch object for the training loop
    torch.save(data, 'graph_data.pt')
    print("\nSuccess! Saved PyTorch graph object to 'graph_data.pt'")

if __name__ == "__main__":
    build_pyg_data()