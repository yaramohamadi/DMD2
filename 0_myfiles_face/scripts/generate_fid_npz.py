import os
import numpy as np
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from main.dhariwal.evaluation_util import compute_statistics_of_path

# Set path to your real dataset
image_path = "/export/datasets/public/diffusion_datasets/adaptation/datasets/fid_folders/Metfaces1336_resized/" 

# Compute Inception features (mean, covariance)
mu, sigma, act = compute_statistics_of_path(image_path)

# Save them into .npz file
np.savez("/export/datasets/public/diffusion_datasets/adaptation/datasets/fid_npz/metfaces.npz", mu=mu, sigma=sigma, act=act)

print("Saved metfaces.npz successfully.")