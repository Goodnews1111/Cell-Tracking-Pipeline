import pandas as pd
import trackpy as tp

# 1. Load generated detection data
df = pd.read_csv('tracking_data.csv')

# 2. Run linker with lower range and adaptive search
linked_df = tp.link_df(
    df, 
    search_range=10,           # Reduced from 20 to 10
    pos_columns=['x', 'y', 'z_slice'], 
    t_column='time', 
    memory=2,
    adaptive_stop=0.5,        # Shrinks search radius if a cluster is too crowded
    adaptive_step=0.95
)

# 3. Rename particle column to track_id
linked_df = linked_df.rename(columns={'particle': 'track_id'})

# 4. Print summary
print("--- Linking Successful ---")
print(f"Total Detections: {len(linked_df)}")
print(f"Total Unique Tracks: {linked_df['track_id'].nunique()}")

# 5. Save output
linked_df.to_csv('linked_tracks.csv', index=False)
print("\nSaved output to 'linked_tracks.csv' successfully!")