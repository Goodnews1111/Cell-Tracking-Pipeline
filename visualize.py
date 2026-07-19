import zarr
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float, restoration, filters
from skimage.morphology import white_tophat, disk

def run_pipeline(file_path):
    print("Loading data...")
    dataset = zarr.open(file_path, mode='r')
    # Using Z-slice 32 as the sample
    image = img_as_float(dataset['0'][0, 32, :, :])

    print("Running Top-Hat background flattening...")
    # 1. Background Flattening: removes uneven lighting
    background = white_tophat(image, footprint=disk(15))

    print("Running NLM Denoising (this may take a moment)...")
    # 2. Extreme Denoising: preserves sharp edges while killing noise
    denoised = restoration.denoise_nl_means(
        background, h=0.08, fast_mode=True, patch_size=5, patch_distance=6
    )

    # 3. Contrast Stretch: ensures optimal intensity range
    final_clean = exposure.rescale_intensity(denoised, out_range=(0, 1))

    print("Running Otsu Segmentation...")
    # 4. Binary Segmentation: creates the precise mask for tracking
    thresh_val = filters.threshold_otsu(final_clean)
    binary_mask = final_clean > thresh_val

    # 5. Visualization: Save the composite results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(final_clean, cmap='gray')
    axes[1].set_title("Extreme Clean")
    axes[1].axis('off')
    
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title("Binary Mask (Otsu)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("visualize_full_pipeline.png")
    print("Pipeline complete. Saved composite to visualize_full_pipeline.png")

if __name__ == "__main__":
    # Ensure this path matches your data structure
    path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"
    run_pipeline(path)