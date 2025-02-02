
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def batch_process_return(detected_faces, img_path,label):
    if detected_faces:
        return {"img_path": img_path, "deteccion": detected_faces, "class": label}
    else:
        return {"img_path": img_path, "deteccion": None, "class": label}


def batch_dataset_face_detection_dlib(dataset, detector, batch_size=32, num_workers=4):
    def process_image(sample):
        img_path, label = sample
        with Image.open(img_path) as img:
            img_array = np.array(img)
            detected_faces = detector(img_array)
        return batch_process_return(detected_faces,img_path, label)

    total_samples = len(dataset.samples)

    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        with tqdm(total=total_samples, desc="Detectando Rostros", unit="image") as pbar:
            for batch_start in range(0, total_samples, batch_size):
                batch = dataset.samples[batch_start:batch_start + batch_size]
                for sample in batch:
                    futures.append(executor.submit(process_image, sample))

                # Collect batch results
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)  # Update progress bar
                futures = []

    return results


def batch_dataset_face_detection_cnn(dataset, detector, batch_size=32, num_workers=4):
    def process_image(sample):
        img_path, label = sample
        detected_faces = detector.detect_faces(img_path)

        return batch_process_return(detected_faces,img_path, label)

    total_samples = len(dataset.samples)

    results = []
    # with Pool(num_workers) as pool:
    #    for batch_start in tqdm(range(0, total_samples, batch_size), desc="Detección de Rostro", unit="batch"):
    #        batch = dataset.samples[batch_start:batch_start + batch_size]
    #        args = [(path, label) for path, label in batch]
    #        results = pool.map(process_image, batch)
    #        results.extend(results)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        with tqdm(total=total_samples, desc="Detectando Rostros", unit="image") as pbar:
            for batch_start in range(0, total_samples, batch_size):
                batch = dataset.samples[batch_start:batch_start + batch_size]
                for sample in batch:
                    futures.append(executor.submit(process_image, sample))

                # Collect batch results
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)  # Update progress bar
                futures = []
                # for batch_start in tqdm(range(0, total_samples, batch_size), desc="Detección de Rostro", unit="batch"):
    #    batch = dataset.samples[batch_start:batch_start + batch_size]
    #    batch_results = [process_image(sample) for sample in batch]
    #    results.extend(batch_results)

    return results


def process_batch_results(results, dataset, detector):
    usable_dataset = [r for r in results if r["deteccion"] is not None]
    failed = [r for r in results if r["deteccion"] is None]
    class_counts = {idx: 0 for idx in range(len(dataset.classes))}
    detected_counts = {idx: 0 for idx in range(len(dataset.classes))}

    for result in results:
        class_idx = result["class"]
        class_counts[class_idx] += 1
        if result["deteccion"] is not None:
            detected_counts[class_idx] += 1

    df_counts = pd.DataFrame({
        "Clase": [dataset.classes[class_idx] for class_idx in class_counts.keys()],
        "Total": list(class_counts.values()),
        "Detectados": [detected_counts[class_idx] for class_idx in class_counts.keys()],
    })

    print(f"Detector: {detector}")
    print(f"Dataset Original: {len(dataset)}")
    print(f"Dataset Usable: {len(usable_dataset)}")
    print(f"Dataset Fallido: {len(failed)}")

    return usable_dataset, df_counts, failed
