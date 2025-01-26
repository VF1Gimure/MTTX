import os
import cv2
import numpy as np
from PIL import Image
from pathos.multiprocessing import Pool
from tqdm import tqdm
import pandas as pd


def calculate_noise(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("L")  # Convert to grayscale
            img_array = np.array(img, dtype=np.uint8)
            high_pass = cv2.Laplacian(img_array, cv2.CV_64F)
            noise = high_pass.std()
            return {"sharpness": high_pass.var(), "noise": noise}
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def batch_sharpness(dataframe, dataset, save_dir, name, batch_size=32, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{name}_metrics.pkl")

    metrics = []
    total_samples = len(dataset.samples)

    with Pool(num_workers) as pool:
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Calculating sharpness", unit="batch"):
            batch = dataset.samples[batch_start:batch_start + batch_size]
            paths = [path for path, _ in batch]
            results = pool.map(calculate_noise, paths)
            metrics.extend(results)

    metric_df = pd.DataFrame(metrics)
    updated_df = pd.concat([dataframe.reset_index(drop=True), metric_df], axis=1)
    updated_df.to_pickle(save_file)

    return updated_df

def calculate_flattened_image_metrics(flattened_img):
    brightness = flattened_img.mean()

    contrast = flattened_img.std()
    hist, _ = np.histogram(flattened_img, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-7))  # Calculate entropy

    return {"brightness": brightness, "contrast": contrast, "entropy": entropy}


def batch_calculate_image_metrics(flattened_df, name, save_dir, batch_size=32, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{name}_metrics.pkl")

    data = []
    total_samples = len(flattened_df)

    with Pool(num_workers) as pool:
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Calculating Metrics", unit="batch"):
            batch = flattened_df.iloc[batch_start:batch_start + batch_size]
            args = [np.array(img) for img in batch["flattened_image"]]
            results = pool.map(calculate_flattened_image_metrics, args)

            for idx, metrics in zip(batch.index, results):
                row = batch.loc[idx].to_dict()
                row.update(metrics)
                data.append(row)

    updated_df = pd.DataFrame(data)
    updated_df.to_pickle(save_file)


def batch_image_shape(dataset, batch_size=32, num_workers=4):
    metrics = []
    total_samples = len(dataset.samples)

    def image_attributes(img_path):
        with Image.open(img_path) as img:
            img_array = np.array(img)
            attributes = {
                "width": img.width,
                "height": img.height,
                "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1,  # Handle grayscale
                "format": img.format,
                "mode": img.mode
            }
            return attributes

    with Pool(num_workers) as pool:
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Adding Attributes", unit="batch"):
            batch = dataset.samples[batch_start:batch_start + batch_size]
            paths = [path for path, _ in batch]
            results = pool.map(image_attributes, paths)
            metrics.extend(results)

    return pd.DataFrame(metrics)
