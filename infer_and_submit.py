import pandas as pd
import torch
import networkx as nx
from train_gnn import CellEdgeClassifier

def main():
    print("--- Loading Trained Model and Data ---")
    try:
        data = torch.load('graph_data.pt', weights_only=False)
        nodes_df = pd.read_csv('tracking_data.csv')
        edges_df = pd.read_csv('graph_edges.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Initialize model and load trained weights
    model = CellEdgeClassifier()
    model.load_state_dict(torch.load('gnn_tracker_weights.pth', weights_only=True))
    model.eval()

    print("--- Running AI Inference ---")
    with torch.no_grad():
        predictions = model(data.x, data.edge_index, data.edge_attr).numpy().flatten()
    
    edges_df['probability'] = predictions
    
    # Filter high-confidence links (>80% confidence)
    threshold = 0.80
    confident_edges = edges_df[edges_df['probability'] > threshold]
    print(f"Filtered down to {len(confident_edges)} high-confidence links out of {len(edges_df)}.")

    print("--- Formatting for Kaggle Competition Schema ---")
    dataset_name = "test_dataset" # Matches test set zarr folder convention
    
    all_rows = []
    global_id = 0

    # 1. Add Node Rows (Cell Detections)
    for _, row in nodes_df.iterrows():
        all_rows.append({
            'id': global_id,
            'dataset': dataset_name,
            'row_type': 'node',
            'node_id': int(row['global_node_id']),
            't': int(row['time']),
            'z': int(row['z_slice']),
            'y': int(row['y']),
            'x': int(row['x']),
            'source_id': -1,
            'target_id': -1
        })
        global_id += 1

    # 2. Add Edge Rows (GNN Track Links)
    for _, row in confident_edges.iterrows():
        all_rows.append({
            'id': global_id,
            'dataset': dataset_name,
            'row_type': 'edge',
            'node_id': -1,
            't': -1,
            'z': -1,
            'y': -1,
            'x': -1,
            'source_id': int(row['source_node']),
            'target_id': int(row['target_node'])
        })
        global_id += 1

    # Create final submission DataFrame with exact column order
    column_order = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    submission_df = pd.DataFrame(all_rows)[column_order]
    
    submission_df.to_csv('submission.csv', index=False)
    print(f"\n🏆 Saved 'submission.csv' with {len(submission_df)} total rows formatted correctly for Kaggle!")

if __name__ == "__main__":
    main()