import zarr
import numpy as np
from skimage import measure, filters, restoration, morphology, segmentation, feature, exposure
from scipy import ndimage as ndi

def check_label_stats(file_path):
    dataset = zarr.open(file_path, mode='r')
    image = dataset['0'][0, 32, :, :] # Sample slice
    
    # --- Reprocess to match your pipeline ---
    background = morphology.white_tophat(image, morphology.disk(15))
    denoised = restoration.denoise_nl_means(background, h=0.08, fast_mode=True, patch_size=5, patch_distance=6)
    final_clean = exposure.rescale_intensity(denoised, out_range=(0, 1))
    binary_mask = final_clean > filters.threshold_otsu(final_clean)
    
    # --- Watershed ---
    distance = ndi.distance_transform_edt(binary_mask)
    coords = feature.peak_local_max(distance, min_distance=15, labels=binary_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=binary_mask)
    
    # --- THE FINAL CHECKS ---
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]
    
    print(f"Total Objects Detected: {len(props)}")
    print(f"Average Cell Area: {np.mean(areas):.2f} pixels")
    print(f"Minimum Cell Area: {np.min(areas)} pixels")
    print(f"Maximum Cell Area: {np.max(areas)} pixels")
    
    # Identifying potential issues
    if np.min(areas) < 10:
        print("⚠️ WARNING: Very small objects detected. You may have noise masquerading as cells.")
    if np.max(areas) > (np.mean(areas) * 3):
        print("⚠️ WARNING: Very large objects detected. Some cells may be merged (Under-segmentation).")

if __name__ == "__main__":
    path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"
    check_label_stats(path)