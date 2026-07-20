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
    """
    divisions = []
    
    # Calculate terminal coordinates for all tracks
    track_ends = df.groupby('track_id').agg(
        end_time=('time', 'max'),
        x_end=('x', 'last'),
        y_end=('y', 'last'),
        z_end=('z_slice', 'last')
    ).reset_index()

    # Calculate initial coordinates for all tracks
    track_starts = df.groupby('track_id').agg(
        start_time=('time', 'min'),
        x_start=('x', 'first'),
        y_start=('y', 'first'),
        z_start=('z_slice', 'first')
    ).reset_index()

    # Link parent terminals to daughter origins
    for _, parent in track_ends.iterrows():
        t_end = parent['end_time']
        p_id = parent['track_id']
        p_pos = np.array([parent['x_end'], parent['y_end'], parent['z_end']])

        # Look for daughters starting at frame t + 1
        candidates = track_starts[track_starts['start_time'] == t_end + 1]

        if len(candidates) >= 2:
            cand_pos = candidates[['x_start', 'y_start', 'z_start']].values
            distances = np.linalg.norm(cand_pos - p_pos, axis=1)

            # Identify daughters within distance threshold
            close_daughters = candidates[distances <= max_daughter_dist]['track_id'].tolist()

            if len(close_daughters) == 2:
                divisions.append({
                    'parent_track_id': p_id,
                    'daughter_1_id': close_daughters[0],
                    'daughter_2_id': close_daughters[1],
                    'division_frame': t_end + 1
                })

    return pd.DataFrame(divisions)

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

    # 2. Detect cell division lineage
    div_df = detect_cell_divisions(cleaned_df, max_daughter_dist=15.0)
    print(f"Detected mitosis events: {len(div_df)}")
    div_df.to_csv('cell_divisions.csv', index=False)

    print("\nAnalysis complete! Saved 'cleaned_tracks.csv' and 'cell_divisions.csv'.")

if __name__ == '__main__':
    main()