import zarr
import matplotlib.pyplot as plt
from skimage import exposure, img_as_float, restoration
from skimage.morphology import white_tophat, disk

# Load data
file_path = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train/6bba_cf35214c.zarr"
dataset = zarr.open(file_path, mode='r')

# Select a mid-slice (Time 0, Z-slice 32)
# Ensure we convert to float for precise processing
image = img_as_float(dataset['0'][0, 32, :, :])

# 1. Background Flattening (Top-Hat)
# This flattens the uneven background lighting
background = white_tophat(image, footprint=disk(15))

# 2. Non-Local Means Denoising (The "Extreme" part)
# This preserves sharp cell edges better than a standard blur
denoised = restoration.denoise_nl_means(
    background, 
    h=0.08, 
    fast_mode=True, 
    patch_size=5, 
    patch_distance=6
)

# 3. Final Contrast Stretch to [0, 1] range
final = exposure.rescale_intensity(denoised, out_range=(0, 1))

# Save the result
plt.imsave("extreme_clean.png", final, cmap='gray')
print("High-precision cleaning complete. Saved to extreme_clean.png")