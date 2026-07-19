import zarr
import pandas as pd
import numpy as np
from skimage import measure, filters, restoration, morphology, segmentation, feature, exposure
from scipy import ndimage as ndi

def get_centroids(file_path, output_csv="tracking_data.csv"):
    dataset = zarr.open(file_path, mode='r')
    tracking_data = []

    print("Processing entire volume for centroids...")
    
    # Iterate through every time frame and Z-slice
    # Adjust ranges based on your dataset size
    for t in range(dataset.shape[0]):
        for z in range(dataset.shape[1]):
            image = dataset[t, z, :, :]
            
            # --- Repeat your proven pipeline ---
            background = morphology.white_tophat(image, morphology.disk(15))
            denoised = restoration.denoise_nl_means(background, h=0.08, fast_mode=True, patch_size=5, patch_distance=6)
            final_clean = exposure.rescale_intensity(denoised, out_range=(0, 1))
            binary_mask = final_clean > filters.threshold_otsu(final_clean)
            
            # Watershed separation
            distance = ndi.distance_transform_edt(binary_mask)
            coords = feature.peak_local_max(distance, min_distance=15, labels=binary_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = segmentation.watershed(-distance, markers, mask=binary_mask)
            
            # --- Extract Centroids ---
            props = measure.regionprops(labels)
            for prop in props:
                y, x = prop.centroid
                tracking_data.append({
                    "time": t,
                    "z_slice": z,
                    "cell_id": prop.label,
                    "x": x,
                    "y": y
                })
        print(f"Processed time frame {t}")

    # Save to CSV
    df = pd.DataFrame(tracking_data)
    df.to_csv(output_csv, index=False)
    print(f"Tracking data saved to {output_csv}")

if __name__ == "__main__":
    path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"
    get_centroids(path)