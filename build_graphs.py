import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def create_candidate_edges(df, max_search_radius=25.0):
    """
    Creates a graph connecting cells in frame t to multiple candidate 
    cells in frame t+1. The GNN will later score these edges.
    """
    edges = []
    
    # Group detections by time
    frames = sorted(df['time'].unique())
    
    for i in range(len(frames) - 1):
        t_current = frames[i]
        t_next = frames[i + 1]
        
        # Get nodes (cells) for current and next frame
        nodes_t = df[df['time'] == t_current].copy()
        nodes_next = df[df['time'] == t_next].copy()
        
        if nodes_t.empty or nodes_next.empty:
            continue
            
        coords_t = nodes_t[['x', 'y', 'z_slice']].values
        coords_next = nodes_next[['x', 'y', 'z_slice']].values
        
        # Build spatial tree for the next frame
        tree_next = cKDTree(coords_next)
        
        # Query points: find up to 5 nearest neighbors within our radius
        distances, indices = tree_next.query(
            coords_t, 
            k=5, 
            distance_upper_bound=max_search_radius
        )
        
        # Compile the graph edges
        for i_t, (dist_array, idx_array) in enumerate(zip(distances, indices)):
            for dist, i_next in zip(dist_array, idx_array):
                # cKDTree returns 'inf' and out-of-bounds indices for empty matches
                if dist != float('inf') and i_next < len(nodes_next):
                    source_id = nodes_t.iloc[i_t]['global_node_id']
                    target_id = nodes_next.iloc[i_next]['global_node_id']
                    
                    edges.append({
                        'time_t': t_current,
                        'source_node': source_id,
                        'target_node': target_id,
                        'distance': dist,
                        'delta_x': nodes_next.iloc[i_next]['x'] - nodes_t.iloc[i_t]['x'],
                        'delta_y': nodes_next.iloc[i_next]['y'] - nodes_t.iloc[i_t]['y'],
                        'delta_z': nodes_next.iloc[i_next]['z_slice'] - nodes_t.iloc[i_t]['z_slice']
                    })
    
    return pd.DataFrame(edges)

def main():
    print("--- Phase 1: Constructing Spatial Graphs ---")
    
    try:
        # We start with the raw detections, not the linked tracks
        df = pd.read_csv('tracking_data.csv')
    except FileNotFoundError:
        print("Error: 'tracking_data.csv' not found. Ensure extraction script ran.")
        return
        
    # 1. Create a unique global ID for every single detection (node) across all frames
    if 'global_node_id' not in df.columns:
        df['global_node_id'] = range(len(df))
        # Save it back so our raw data permanently has these node IDs
        df.to_csv('tracking_data.csv', index=False)
        
    print(f"Loaded {len(df)} total biological cell detections.")
    
    # 2. Build candidate graph edges (Generous 25px radius to capture all possibilities)
    edge_df = create_candidate_edges(df, max_search_radius=25.0)
    
    print(f"Generated {len(edge_df)} candidate edges.")
    
    # 3. Save the graph structure
    edge_df.to_csv('graph_edges.csv', index=False)
    print("Saved neural network graph structure to 'graph_edges.csv'.")

if __name__ == "__main__":
    main()