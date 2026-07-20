import pandas as pd
import numpy as np

def clean_noise_tracks(df, min_length=4):
    """Filter out short, transient tracks (flickers/noise)."""
    track_counts = df['track_id'].value_counts()
    valid_ids = track_counts[track_counts >= min_length].index
    return df[df['track_id'].isin(valid_ids)].copy()

def detect_cell_divisions(df, max_daughter_dist=15.0):
    """
    Detect mitosis: parent track ending at frame t where two daughter 
    tracks emerge at frame t+1 within spatial proximity.
    Includes distance-based tie-breaking for dense clusters.
    """
    divisions = []
    
    # Track endpoints
    track_ends = df.groupby('track_id').agg(
        end_time=('time', 'max'),
        x_end=('x', 'last'),
        y_end=('y', 'last'),
        z_end=('z_slice', 'last')
    ).reset_index()

    # Track start points
    track_starts = df.groupby('track_id').agg(
        start_time=('time', 'min'),
        x_start=('x', 'first'),
        y_start=('y', 'first'),
        z_start=('z_slice', 'first')
    ).reset_index()

    for _, parent in track_ends.iterrows():
        t_end = parent['end_time']
        p_id = parent['track_id']
        p_pos = np.array([parent['x_end'], parent['y_end'], parent['z_end']])

        # Find daughters emerging at frame t + 1
        candidates = track_starts[track_starts['start_time'] == t_end + 1]

        if len(candidates) >= 2:
            cand_pos = candidates[['x_start', 'y_start', 'z_start']].values
            distances = np.linalg.norm(cand_pos - p_pos, axis=1)

            close_mask = distances <= max_daughter_dist
            close_candidates = candidates[close_mask]
            close_dists = distances[close_mask]

            if len(close_candidates) == 2:
                d_ids = close_candidates['track_id'].tolist()
                divisions.append({
                    'parent_track_id': p_id,
                    'daughter_1_id': d_ids[0],
                    'daughter_2_id': d_ids[1],
                    'division_frame': t_end + 1,
                    'avg_distance': close_dists.mean()
                })

    div_df = pd.DataFrame(divisions)
    
    # Resolve multi-parent collisions: keep the single parent with the smallest distance
    if not div_df.empty:
        div_df = (div_df.sort_values('avg_distance')
                        .groupby(['daughter_1_id', 'daughter_2_id'])
                        .first()
                        .reset_index())
        div_df = div_df.drop(columns=['avg_distance'])

    return div_df

def main():
    print("--- Loading Linked Tracks ---")
    try:
        df = pd.read_csv('linked_tracks.csv')
    except FileNotFoundError:
        print("Error: 'linked_tracks.csv' not found. Run linker.py first.")
        return

    print(f"Total raw tracks: {df['track_id'].nunique()}")

    # 1. Clean noisy tracks
    cleaned_df = clean_noise_tracks(df, min_length=4)
    print(f"Cleaned tracks (length >= 4): {cleaned_df['track_id'].nunique()}")
    cleaned_df.to_csv('cleaned_tracks.csv', index=False)

    # 2. Detect cell division lineage with collision resolution
    div_df = detect_cell_divisions(cleaned_df, max_daughter_dist=15.0)
    print(f"Detected true mitosis events (conflict-resolved): {len(div_df)}")
    div_df.to_csv('cell_divisions.csv', index=False)

    print("\nAnalysis complete! Saved 'cleaned_tracks.csv' and 'cell_divisions.csv'.")

if __name__ == '__main__':
    main()