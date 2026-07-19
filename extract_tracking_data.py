import zarr
import pandas as pd
import numpy as np
from skimage import measure, filters, restoration, morphology, segmentation, feature, exposure
from scipy import ndimage as ndi

def get_centroids(file_path, output_csv="tracking_data.csv"):
    # Open the Zarr group
    dataset = zarr.open(file_path, mode='r')
    
    # Point to the actual data array inside the group
    data_array = dataset['0'] 
    
    tracking_data = []

    print(f"Processing volume with shape: {data_array.shape}")
    print(f"Targeting file: {file_path}")
    
    # Use data_array to iterate through dimensions
    for t in range(data_array.shape[0]):
        for z in range(data_array.shape[1]):
            # Access the image slice from the array
            image = data_array[t, z, :, :]
            
            # --- The Robust Pipeline ---
            background = morphology.white_tophat(image, morphology.disk(15))
            denoised = restoration.denoise_nl_means(background, h=0.08, fast_mode=True, patch_size=5, patch_distance=6)
            final_clean = exposure.rescale_intensity(denoised, out_range=(0, 1))
            binary_mask = final_clean > filters.threshold_otsu(final_clean)
            
            # Watershed separation (to handle cell division and touching cells)
            distance = ndi.distance_transform_edt(binary_mask)
            coords = feature.peak_local_max(distance, min_distance=15, labels=binary_mask)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True
            markers, _ = ndi.label(mask)
            labels = segmentation.watershed(-distance, markers, mask=binary_mask)
            
            # --- Extract Spatial Coordinates ---
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

    # Save the mathematical structure to CSV
    df = pd.DataFrame(tracking_data)
    df.to_csv(output_csv, index=False)
    print(f"Tracking data successfully generated and saved to {output_csv}")

if __name__ == "__main__":
    # NEW TARGET: Testing a different file for the 71% Private Set Robustness Check
    path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/44b6_2f31fc2f.zarr"
    get_centroids(path)