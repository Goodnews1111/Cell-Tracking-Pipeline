import zarr
import matplotlib.pyplot as plt
import numpy as np

# Path to one of the files we just identified
file_path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"

def visualize_slice(path):
    dataset = zarr.open(path, mode='r')
    # Let's take the first time frame (0) and the middle Z-slice (32)
    # The shape is (100, 64, 256, 256)
    img_slice = dataset['0'][0, 32, :, :]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_slice, cmap='gray')
    plt.title(f"Z-slice 32 of {path.split('/')[-1]}")
    plt.savefig("sample_frame.png")
    print("Saved visualization to sample_frame.png")

if __name__ == "__main__":
    visualize_slice(file_path)