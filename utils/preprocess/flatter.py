
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import os
from pathos.multiprocessing import Pool

def process_image_flat(args):
    path, label, classes = args
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            flattened_img = np.array(img).flatten()
            return {
                "img_path": path,
                "class": classes[label],
                "class_index": label,
                "flattened_image": flattened_img.tolist()
            }
    except Exception as e:
        print(e)
        return None


def batch_flatten_img(dataset, save_dir, file='_flatten.pkl', batch_size=32, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, file)

    data = []
    total_samples = len(dataset.samples)

    with Pool(num_workers) as pool:
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Flattening Images", unit="batch"):
            batch = dataset.samples[batch_start:batch_start + batch_size]
            args = [(path, label, dataset.classes) for path, label in batch]
            results = pool.map(process_image_flat, args)
            data.extend([res for res in results if res is not None])

    df = pd.DataFrame(data)
    df.to_pickle(save_file)