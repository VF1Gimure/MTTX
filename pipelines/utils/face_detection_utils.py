import torch
from torchvision.transforms.functional import to_pil_image
from facenet_pytorch import MTCNN
from pipelines.utils.data_utils import denormalize
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
import dlib
from collections import defaultdict
from pipelines.utils.data_utils import denormalize
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim  # SSIM for image similarity
import re

def extract_mtcnn_box_s(detector, idx, tensor, mean, std):
    """Extracts face bounding box using MTCNN from a pre-loaded tensor."""
    try:
        # Denormalize tensor
        tensor_denorm = denormalize(tensor, mean, std)

        # Convert back to PIL image
        image_pil = to_pil_image(tensor_denorm)

        # Detect face box
        boxes, _ = detector.detect(image_pil)

        if boxes is None or len(boxes) == 0:
            return idx, None  # No face detected

        return idx, boxes[0]  # Store only bounding box

    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return idx, None  # Handle image errors


def rm_similar_keep_boxes_mtcnn_s(detector, loaded_data, mean, std, threshold=5.0, num_workers=8):
    """
    Extracts MTCNN bounding boxes and removes similar images.

    Args:
        detector (MTCNN): Face detection model.
        loaded_data (dict): Dataset containing idxs, tensors, labels, and file_paths.
        mean (torch.Tensor): Normalization mean values.
        std (torch.Tensor): Normalization std values.
        threshold (float): Similarity threshold for removing duplicates.
        num_workers (int): Number of parallel workers.

    Returns:
        tuple: (unique_indices, unique_boxes, removed_indices)
    """
    removed_indices = []
    unique_indices = []
    unique_boxes = []

    # Create list of (idx, tensor) for all images
    all_entries = list(zip(*loaded_data.values()))  # Unpack idxs, tensors, labels, file_paths

    # Process all images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: extract_mtcnn_box_s(detector, entry[0], entry[1], mean, std), all_entries),
            total=len(all_entries),
            desc="Extracting Bounding Boxes for All Images"
        ))

    last_box = None

    for idx, box in results:
        if box is None:
            removed_indices.append(idx)
            continue

        if last_box is not None:
            distances = np.linalg.norm(box - last_box, axis=0)
            mean_distance = np.mean(distances)

            if mean_distance < threshold:
                removed_indices.append(idx)
                continue  # Skip this image

        # Store unique metadata
        last_box = box
        unique_indices.append(idx)
        unique_boxes.append(box)

    return unique_indices, unique_boxes, removed_indices


def extract_mtcnn_boxes_s(detector, loaded_data, mean, std, num_workers=8):

    unique_indices = loaded_data["idxs"]
    unique_images = loaded_data["tensors"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: extract_mtcnn_box_s(detector, entry[0], entry[1], mean, std),
                         zip(unique_indices, unique_images)),
            total=len(unique_images),
            desc="Extracting Bounding Boxes for Unique Images"
        ))

    filter_boxes = {idx: box for idx, box in results if box is not None}

    return filter_boxes


def tensor_to_numpy(img_tensor, mean, std):
    img_tensor = denormalize(img_tensor, mean, std)
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) → (H, W, C)

    img_np = (img_np * 255).astype(np.uint8)  # Convert to uint8 (0-255)
    return img_np


def skip_frames(images, frame_skip):
    if frame_skip <= 1:  # No skipping needed
        return images

    # Sort images by frame number or timestamp
    def extract_frame_number(path):
        match = re.search(r'(\d+)', path)
        return int(match.group(0)) if match else float('inf')  # Ensure we sort by number

    images_sorted = sorted(images, key=lambda x: extract_frame_number(x[2]))  # Sort by frame number

    # Now skip frames correctly
    filtered_images = []
    last_kept_frame = None

    for i, (idx, tensor, path) in enumerate(images_sorted):
        frame_number = extract_frame_number(path)

        if last_kept_frame is None or (frame_number - last_kept_frame) >= frame_skip:
            filtered_images.append((idx, tensor, path))
            last_kept_frame = frame_number  # Update last kept frame

    return filtered_images


def extract_mtcnn_box(detector, idx, tensor, mean, std):
    """Extracts face bounding box and landmarks using MTCNN from a pre-loaded tensor."""
    # Denormalize tensor
    tensor_denorm = denormalize(tensor, mean, std)

    # Convert back to PIL image
    image_pil = to_pil_image(tensor_denorm)

    # Detect face box and landmarks
    boxes, _, landmarks = detector.detect(image_pil, landmarks=True)

    if boxes is None or landmarks is None:
        return (idx, None, None)  # No face detected

    return (idx, boxes[0], landmarks[0])  # Store bounding box and landmarks


def extract_mtcnn_boxes(detector, loaded_data, mean, std, num_workers=8, frame_skip=2):
    """
    Extracts MTCNN bounding boxes and landmarks while skipping frames.

    Returns:
        - filter_boxes_r (dict): {idx: box}
        - filter_landmarks_r (dict): {idx: landmarks}
    """
    grouped_images = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, tensor, label, path in zip(
            loaded_data["idxs"], loaded_data["tensors"], loaded_data["labels"], loaded_data["file_paths"]
    ):
        size_key = tuple(tensor.shape[-2:])
        person_id = "timestamp" if "_#" in path and len(path.split("_#")[-1]) > 3 else "number"
        grouped_images[label][person_id][size_key].append((idx, tensor, path))

    # **Prepare filtered images BEFORE processing**
    filtered_images = []
    for label, persons in grouped_images.items():
        for person_id, sizes in persons.items():
            for size, images in sizes.items():
                filtered_images.extend(skip_frames(images, frame_skip))

    total_images = len(filtered_images)

    # **Process images in parallel with MTCNN**
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: extract_mtcnn_box(detector, entry[0], entry[1], mean, std), filtered_images),
            total=total_images,
            desc="Extracting MTCNN Boxes & Landmarks",
            dynamic_ncols=True
        ))
    filter_boxes_r = {}
    filter_landmarks_r = {}
    for idx, box, landmarks in results:
        if box is None or landmarks is None:
            continue
        filter_boxes_r[idx] = box
        filter_landmarks_r[idx] = landmarks

    return filter_boxes_r, filter_landmarks_r


def image_similarity(img1_tensor, img2_tensor, mean, std):
    img1 = tensor_to_numpy(img1_tensor, mean, std)
    img2 = tensor_to_numpy(img2_tensor, mean, std)

    min_dim = min(img1.shape[0], img1.shape[1])  # Smallest dimension
    win_size = min(7, min_dim)  # Set win_size to at most 7 or the smallest image dimension

    # Compute SSIM with updated parameters
    ssim_score = ssim(img1, img2, win_size=win_size, channel_axis=-1)
    return ssim_score


def remove_similar_images(loaded_data, mean, std, threshold=0.9):
    """
    Removes similar images within the same person’s video based on SSIM.

    Args:
        loaded_data (dict): Dataset containing idxs, tensors, labels, and file_paths.
        mean (torch.Tensor): Normalization mean values.
        std (torch.Tensor): Normalization std values.
        threshold (float): SSIM similarity threshold (default=0.9).

    Returns:
        dict: Filtered dataset with unique images.
    """
    unique_indices = []
    unique_images = []
    unique_labels = []
    unique_paths = []
    removed_indices = []
    total_removed = 0  # Track total removed images

    tensors, labels, file_paths = (
        loaded_data["tensors"],
        loaded_data["labels"],
        loaded_data["file_paths"]
    )

    # Group images by class, person ID, and size
    grouped_images = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, (tensor, label, path) in enumerate(zip(tensors, labels, file_paths)):
        size_key = tuple(tensor.shape[-2:])  # (Height, Width)
        person_id = "timestamp" if "_#" in path and len(path.split("_#")[-1]) > 3 else "number"
        grouped_images[label][person_id][size_key].append((idx, tensor, path))

    # Total images for tqdm progress bar
    total_images = sum(
        len(images) for persons in grouped_images.values() for sizes in persons.values() for images in sizes.values())

    with tqdm(total=total_images, desc="Removing Similar Images", dynamic_ncols=True) as pbar:
        for label, persons in grouped_images.items():
            for person_id, sizes in persons.items():
                for size, images in sizes.items():
                    class_images = {}  # Stores unique images for this person & size

                    for idx, img, path in images:
                        similar = False

                        for u_idx, u_img in class_images.items():
                            similarity = image_similarity(img, u_img, mean, std)
                            if similarity > threshold:
                                removed_indices.append(idx)
                                total_removed += 1  # Increment removed count
                                similar = True
                                break  # Stop checking if a match is found

                        if not similar:
                            class_images[idx] = img

                        # Update tqdm after each image is processed
                        pbar.update(1)
                        pbar.set_postfix(removed=total_removed)

                    # Store only unique images
                    for idx, img in class_images.items():
                        unique_indices.append(idx)
                        unique_images.append(img)
                        unique_labels.append(labels[idx])
                        unique_paths.append(file_paths[idx])

    return {
        "idxs": unique_indices,
        "tensors": unique_images,
        "labels": unique_labels,
        "file_paths": unique_paths,
        "removed_indices": removed_indices
    }


def extract_dlib_landmarks(loaded_data, mean, std, predictor, num_workers=8):
    """
    Extracts Dlib 68 facial landmarks using MTCNN bounding boxes from `loaded_data`.

    Args:
        loaded_data (dict): Dataset containing tensors and bounding boxes.
        mean (torch.Tensor): Normalization mean.
        std (torch.Tensor): Normalization std.
        predictor (dlib.shape_predictor): Dlib 68-landmark model.
        num_workers (int): Number of parallel workers.

    Returns:
        dict: Updated dataset with Dlib landmarks.
    """

    def extract_landmarks(idx, tensor, box):
        """Extracts Dlib landmarks using MTCNN bounding box, handling normalization."""
        x_min, y_min, x_max, y_max = map(int, box)

        # Step 1: **Denormalize the tensor**
        mean_gray = mean.mean()  # Use a single mean for grayscale
        std_gray = std.mean()
        tensor_denorm = tensor * std_gray + mean_gray  # Undo normalization

        # Step 2: **Convert to NumPy (No PIL)**
        img_np = (tensor_denorm.squeeze(0).cpu().numpy() * 255).astype(np.uint8)  # Shape: (H, W)

        # Step 3: **Crop face region**
        face_region = img_np[y_min:y_max, x_min:x_max]

        # Step 4: **Pass directly to Dlib (No PIL conversion)**
        dlib_rect = dlib.rectangle(0, 0, face_region.shape[1], face_region.shape[0])
        shape = predictor(face_region, dlib_rect)

        # Step 5: **Adjust landmarks back to original image coordinates**
        landmarks = [(p.x + x_min, p.y + y_min) for p in shape.parts()]

        return idx, landmarks, dlib_rect

    # Prepare entries from `loaded_data`
    all_entries = list(zip(loaded_data["idxs"], loaded_data["tensors"], loaded_data["boxes"]))

    # Extract landmarks in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: extract_landmarks(entry[0], entry[1], entry[2]), all_entries),
            total=len(all_entries), desc="Extracting 68 Dlib Landmarks"
        ))

    # Store results
    updated_data = loaded_data.copy()
    updated_data["landmarks"] = [landmarks for _, landmarks, _ in results]
    updated_data["dlib_rects"] = [rect for _, _, rect in results]

    return updated_data


def calculate_face_angle(landmarks):
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]

    # Distance from nose to each eye
    left_dist = np.linalg.norm(np.array(left_eye) - np.array(nose))
    right_dist = np.linalg.norm(np.array(right_eye) - np.array(nose))

    # Compute relative difference (yaw estimation)
    angle = np.degrees(np.arctan2(right_dist - left_dist, left_dist + right_dist))
    print(angle)

    return angle


def calculate_face_angle_box(box, image_size):
    img_center_x = image_size[1] / 2  # Get image width center
    face_center_x = (box[0] + box[2]) / 2  # Box center X-coordinate

    deviation = face_center_x - img_center_x  # How far from the center
    angle = (deviation / img_center_x) * 90  # Normalize to a 90° scale
    print(angle)
    return angle  # Positive = Right, Negative = Left


def calculate_face_angle_abs(box, image_size):
    """
    Calculates the face's yaw angle using bounding box edges relative to the image center.

    Args:
        box (array-like): Bounding box (x_min, y_min, x_max, y_max).
        image_size (tuple): (Height, Width) of the image.

    Returns:
        float: Yaw angle (degrees) based on bounding box position.
    """
    img_center = np.array([image_size[1] / 2, image_size[0] / 2])  # (center_x, center_y)

    # Midpoint of the bounding box
    face_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])  # (center_x, center_y)

    # Compute vector from image center to face center
    face_vector = face_center - img_center

    # Reference vector: perfectly centered face (horizontal reference)
    ref_vector = np.array([1, 0])  # (Rightward direction)

    # Compute yaw angle using atan2 (preserves negative values)
    angle_rad = np.arctan2(face_vector[1], face_vector[0])  # Yaw angle in radians
    angle_deg = np.degrees(angle_rad)
    #print(angle_deg)
    return angle_deg


from tqdm import tqdm


def filter_profile_faces(filter_landmarks, angle_threshold=45):
    """
    Filters out extreme profile faces (> ±45° yaw deviation) using MTCNN landmarks.

    Args:
        filter_landmarks (dict): {idx: landmarks} from MTCNN.
        angle_threshold (int): Maximum allowed face deviation from center.

    Returns:
        dict: Filtered {idx: landmarks} containing only forward-facing images.
        list: Removed indices for discarded profile faces.
    """
    filtered_landmarks = {}
    removed_indices = []

    face_angles = {}

    with tqdm(total=len(filter_landmarks), desc="Calculando Ángulos de Rostro", dynamic_ncols=True) as pbar:
        for idx, landmarks in filter_landmarks.items():
            face_angles[idx] = calculate_face_angle(landmarks)
            pbar.update(1)

    with tqdm(total=len(face_angles), desc="Filtrando", dynamic_ncols=True) as pbar:
        for idx, angle in face_angles.items():
            if abs(angle) >= angle_threshold:  # If too far from center
                removed_indices.append(idx)  # Discard extreme profile faces
            else:
                filtered_landmarks[idx] = filter_landmarks[idx]  # Keep valid faces
            pbar.update(1)

    print(f"{len(removed_indices)} Frames Removidos.")

    return filtered_landmarks, removed_indices


def filter_profile_faces_box(filter_boxes, image_sizes, angle_threshold=45):
    """
    Calculates yaw angles for all detected faces and filters extreme profiles.

    Args:
        filter_boxes (dict): {idx: box} mapping from MTCNN detection.
        image_sizes (dict): {idx: (H, W)} mapping for each image.
        angle_threshold (int): Maximum allowed face deviation from center.

    Returns:
        dict: {idx: box} containing only forward-facing images.
        dict: {idx: angle} for all detected images.
        dict: {idx: angle} for discarded profile images.
    """
    filtered_boxes = {}
    face_angles = {}
    removed_indices = {}
    filtered_angles = {}
    with tqdm(total=len(filter_boxes), desc="Calculando Ángulos de Rostro", dynamic_ncols=True) as pbar:
        for idx, box in filter_boxes.items():
            angle = calculate_face_angle_abs(box, image_sizes[idx])
            face_angles[idx] = angle  # Store all angles
            pbar.update(1)

    with tqdm(total=len(face_angles), desc="Filtrando", dynamic_ncols=True) as pbar:
        for idx, angle in face_angles.items():
            if abs(angle) > angle_threshold:
                removed_indices[idx] = angle  # Save removed images with their angle
            else:
                filtered_boxes[idx] = filter_boxes[idx]  # Keep valid faces
                filtered_angles[idx] = angle
            pbar.update(1)

    print(f"{len(removed_indices)} Frames Removidos.")

    return filtered_boxes, filtered_angles, removed_indices