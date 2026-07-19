import zarr
import os

# Pointing directly to where Kaggle mounts the data in the cloud
data_dir = "/kaggle/input/biohub-cell-tracking-during-development/train"

def inspect_zarr(file_path):
    print(f"--- Inspecting: {file_path} ---")
    try:
        dataset = zarr.open(file_path, mode='r')
        if isinstance(dataset, zarr.hierarchy.Group):
            print("Type: Zarr Group")
            for key in dataset.array_keys():
                arr = dataset[key]
                print(f" - {key}: Shape {arr.shape}, Type {arr.dtype}")
        else:
            print(f"Type: Zarr Array | Shape: {dataset.shape}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    if os.path.exists(data_dir):
        print("Scanning Kaggle data directory...")
        for item in os.listdir(data_dir):
            if item.endswith(".zarr"):
                inspect_zarr(os.path.join(data_dir, item))
    else:
        print(f"Directory not found: {data_dir}")
        print("This is normal locally! This script is ready to run once pushed to Kaggle.")