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

    # Initialize model and load your highly-trained weights
    model = CellEdgeClassifier()
    model.load_state_dict(torch.load('gnn_tracker_weights.pth', weights_only=True))
    model.eval() # Set to evaluation mode

    print("--- Running AI Inference ---")
    with torch.no_grad():
        # Flatten the predictions into a simple 1D array
        predictions = model(data.x, data.edge_index, data.edge_attr).numpy().flatten()
    
    edges_df['probability'] = predictions
    
    # AI Filtering: Keep only edges the model is >80% confident in
    threshold = 0.80
    confident_edges = edges_df[edges_df['probability'] > threshold]
    print(f"Filtered down to {len(confident_edges)} high-confidence links out of {len(edges_df)}.")

    print("--- Reconstructing Trajectories ---")
    # Build a mathematical graph to connect the dots
    G = nx.Graph()
    G.add_edges_from(zip(confident_edges['source_node'], confident_edges['target_node']))
    
    # Every connected component is a continuous cell track
    track_mapping = {}
    for track_id, component in enumerate(nx.connected_components(G)):
        for node in component:
            track_mapping[node] = track_id

    # Apply the AI-generated Track IDs back to the original cell coordinates
    nodes_df['track_id'] = nodes_df['global_node_id'].map(track_mapping)
    
    # Drop false-positive noise detections that the AI refused to link
    final_submission = nodes_df.dropna(subset=['track_id']).copy()
    final_submission['track_id'] = final_submission['track_id'].astype(int)

    print(f"Total Unique Cells Tracked: {final_submission['track_id'].nunique()}")

    print("--- Formatting for Kaggle Leaderboard ---")
    # Isolate only the exact columns the competition judges evaluate
    submission_cols = ['time', 'x', 'y', 'z_slice', 'track_id']
    final_submission = final_submission[submission_cols]
    
    final_submission.to_csv('submission.csv', index=False)
    print("\n🏆 BOOM! Saved 'submission.csv'. Ready to dominate the leaderboard!")

if __name__ == "__main__":
    main()