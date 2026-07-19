import zarr
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import exposure, img_as_float, restoration, filters, segmentation, feature
from skimage.morphology import white_tophat, disk

def run_pipeline(file_path):
    print("Loading data...")
    dataset = zarr.open(file_path, mode='r')
    # Using Z-slice 32 as the sample
    image = img_as_float(dataset['0'][0, 32, :, :])

    print("Running Preprocessing (Top-Hat + NLM)...")
    # 1. Background Flattening
    background = white_tophat(image, footprint=disk(15))
    
    # 2. Extreme Denoising
    denoised = restoration.denoise_nl_means(
        background, h=0.08, fast_mode=True, patch_size=5, patch_distance=6
    )
    final_clean = exposure.rescale_intensity(denoised, out_range=(0, 1))

    print("Running Segmentation & Watershed Separation...")
    # 3. Binary Segmentation
    thresh_val = filters.threshold_otsu(final_clean)
    binary_mask = final_clean > thresh_val

    # 4. Watershed Separation
    # Calculate distance to background
    distance = ndi.distance_transform_edt(binary_mask)
    # Find peaks (cell centers)
    coords = feature.peak_local_max(distance, min_distance=15, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    # Perform watershed
    labels = segmentation.watershed(-distance, markers, mask=binary_mask)

    print("Generating visualization...")
    # 5. Visualization: 4-panel comparison
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    
    axes[1].imshow(final_clean, cmap='gray')
    axes[1].set_title("Extreme Clean")
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title("Binary Mask")
    
    axes[3].imshow(labels, cmap='nipy_spectral')
    axes[3].set_title("Watershed Labels")
    
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("visualize_full_pipeline.png")
    print("Pipeline complete. Saved composite to visualize_full_pipeline.png")

if __name__ == "__main__":
    path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"
    run_pipeline(path)