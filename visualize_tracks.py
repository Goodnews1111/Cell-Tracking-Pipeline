import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("--- Loading AI-Generated Tracks ---")
    try:
        df = pd.read_csv('submission.csv')
    except FileNotFoundError:
        print("Error: 'submission.csv' not found. Run infer_and_submit.py first.")
        return
    
    # Find the 5 longest continuous tracks to plot
    track_lengths = df['track_id'].value_counts()
    top_tracks = track_lengths.head(5).index
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for tid in top_tracks:
        # Sort by time to draw the path in the correct chronological order
        track_data = df[df['track_id'] == tid].sort_values('time')
        
        ax.plot(track_data['x'], 
                track_data['y'], 
                track_data['z_slice'], 
                marker='o', 
                linewidth=2,
                markersize=4,
                label=f'Track {tid}')
        
    ax.set_xlabel('X Coordinate (px)')
    ax.set_ylabel('Y Coordinate (px)')
    ax.set_zlabel('Z Slice (Depth)')
    ax.set_title('Top 5 GNN-Predicted Cell Trajectories', fontsize=14)
    plt.legend()
    
    plt.savefig('gnn_3d_visualization.png')
    print("Success! Saved 3D track plot to 'gnn_3d_visualization.png'")

if __name__ == '__main__':
    main()