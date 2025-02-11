from collections import defaultdict

from torchvision import datasets
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import dlib
from torch.utils.data import Dataset
import torch.nn.functional as F


def load_data_image_folder(dataset_path, transformer, num_workers):
    i_dataset = datasets.ImageFolder(dataset_path, transform=transformer)

    # Extract data in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda idx: (idx, i_dataset[idx][0], i_dataset[idx][1], i_dataset.imgs[idx][0]),
                         range(len(i_dataset))),
            total=len(i_dataset),
            desc="Extracting Tensors, Targets, and File Paths..."
        ))

    # Unpack results
    idxs, tensors, labels, file_paths = zip(*results)

    return {
        "idxs": list(idxs),  # Store original dataset indices
        "tensors": list(tensors),
        "labels": list(labels),
        "file_paths": list(file_paths)
    }



def denormalize(tensor, mean, std):
    mean = mean.view(3, 1, 1)  # Reshape for broadcasting
    std = std.view(3, 1, 1)  # Reshape for broadcasting
    return tensor * std + mean  # Undo normalization

def filter_loaded_data(loaded_data, filter_boxes):
    """
    Filters tensors, labels, file_paths based on `filter_boxes`
    while preserving original `idxs`.

    Args:
        loaded_data (dict): Dataset with explicit `idxs`.
        filter_boxes (dict): Dictionary {idx: box} with unique images.

    Returns:
        dict: Filtered dataset with consistent IDs.
    """
    # Convert `loaded_data` into a dictionary mapping idx -> (tensor, label, filepath)
    data_dict = {
        idx: (tensor, label, filepath)
        for idx, tensor, label, filepath in zip(
            loaded_data["idxs"], loaded_data["tensors"], loaded_data["labels"], loaded_data["file_paths"]
        )
    }

    # Keep only entries where idx is in `filter_boxes`
    filtered_data = {
        "idxs": list(filter_boxes.keys()),
        "tensors": [data_dict[idx][0] for idx in filter_boxes],
        "labels": [data_dict[idx][1] for idx in filter_boxes],
        "file_paths": [data_dict[idx][2] for idx in filter_boxes],
        "boxes": list(filter_boxes.values()),  # Get boxes directly
    }

    return filtered_data


def convert_to_dlib_shape(landmarks, bbox):
    """
    Convert a list of (x, y) tuples into a dlib full_object_detection shape object.

    Parameters:
    - landmarks: List of tuples [(x1, y1), (x2, y2), ...]
    - bbox: dlib.rectangle of the face bounding box (needed for dlib structure)

    Returns:
    - dlib.full_object_detection object
    """
    shape = dlib.full_object_detection(bbox, [dlib.point(x, y) for (x, y) in landmarks])
    return shape



def padding_crop_landmark_adjust(tensor, box, landmarks, padding=5):
    """
    Args:
        tensor (torch.Tensor): Grayscale image tensor of shape (1, H, W).
        box (tuple): Bounding box (x_min, y_min, x_max, y_max).
        landmarks (list): List of (x, y) landmark coordinates.
        padding (int): Extra pixels to add around the bounding box.

    Returns:
        torch.Tensor: Cropped face tensor.
        list: Adjusted landmarks for the cropped face.
        tuple: Adjusted bounding box after cropping (x_min, y_min, x_max, y_max).
    """
    _, H, W = tensor.shape  # Get image dimensions

    # Extract bounding box
    x_min, y_min, x_max, y_max = map(int, box)

    # Apply padding while staying within image bounds
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(W, x_max + padding)
    y_max = min(H, y_max + padding)

    cropped_tensor = tensor[:, y_min:y_max, x_min:x_max]  # Preserve grayscale format (1, H, W)

    adjusted_landmarks = [(x - x_min, y - y_min) for (x, y) in landmarks]

    return cropped_tensor, adjusted_landmarks, (x_min, y_min, x_max, y_max)


def crop_faces(loaded_data, padding=5, num_workers=8):
    """
    Args:
        loaded_data (dict): Dataset with tensors, bounding boxes, and landmarks.
        padding (int): Extra pixels to add around the bounding box.
        num_workers (int): Number of threads.

    Returns:
        dict: Updated dataset with cropped tensors and adjusted landmarks.
    """

    def crop_single(idx, tensor, box, landmarks):
        """Crops a single face and adjusts landmarks."""
        cropped_tensor, adjusted_landmarks, new_box = padding_crop_landmark_adjust(
            tensor, box, landmarks, padding
        )
        return idx, cropped_tensor, adjusted_landmarks

    # Prepare entries
    all_entries = list(zip(loaded_data["idxs"], loaded_data["tensors"], loaded_data["boxes"], loaded_data["landmarks"]))

    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: crop_single(*entry), all_entries),
            total=len(all_entries), desc="Cropping Faces"
        ))

    # Store updated cropped tensors and landmarks
    updated_data = {
        "idxs": [idx for idx, _, _ in results],
        "tensors": [crop for _, crop, _ in results],
        "landmarks": [lm for _, _, lm in results],
        "labels": loaded_data["labels"],  # Keep labels unchanged
    }

    return updated_data


def resize_faces(loaded_data, target_size=(512, 512), num_workers=8):
    """
    Args:
        loaded_data (dict): Dataset with cropped tensors and updated landmarks.
        target_size (tuple): Target size (H, W).
        num_workers (int): Number of threads.

    Returns:
        dict: Updated dataset with only idxs, tensors, landmarks, and labels.
    """

    def resize_single(idx, tensor, landmarks):
        """Resizes a single image and adjusts landmarks proportionally."""
        orig_h, orig_w = tensor.shape[1:]  # Get original cropped size
        target_h, target_w = target_size

        # 1️⃣ **Resize Tensor Using Bicubic**
        resized_tensor = F.interpolate(
            tensor.unsqueeze(0), size=target_size, mode="bicubic", align_corners=False
        ).squeeze(0)

        # 2️⃣ **Recalculate Landmarks**
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        adjusted_landmarks = [(int(x * scale_x), int(y * scale_y)) for (x, y) in landmarks]

        return idx, resized_tensor, adjusted_landmarks

    # Prepare entries
    all_entries = list(zip(loaded_data["idxs"], loaded_data["tensors"], loaded_data["landmarks"]))

    # Process in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: resize_single(*entry), all_entries),
            total=len(all_entries), desc="Resizing Faces"
        ))

    # Store only required data
    updated_data = {
        "idxs": [idx for idx, _, _ in results],
        "tensors": [resize for _, resize, _ in results],
        "landmarks": [lm for _, _, lm in results],
        "labels": loaded_data["labels"],  # Keep labels unchanged
    }

    return updated_data


def filter_redundant_angles(filtered_boxes, face_angles, loaded_data, angle_similarity_threshold=5):
    # Group by class and person (same as before)
    grouped_faces = defaultdict(lambda: defaultdict(list))

    for idx in filtered_boxes:
        label = loaded_data["labels"][loaded_data["idxs"].index(idx)]
        path = loaded_data["file_paths"][loaded_data["idxs"].index(idx)]
        person_id = path.split("_")[0]
        grouped_faces[label][person_id].append((idx, face_angles[idx]))

    # New filtered results
    final_boxes = {}
    final_angles = {}
    removed_indices = []

    with tqdm(total=len(grouped_faces), desc="Filtering Redundant Angles", dynamic_ncols=True) as pbar:
        for label, persons in grouped_faces.items():
            for person_id, faces in persons.items():
                faces.sort(key=lambda x: x[1])  # Sort by angle for better filtering

                selected_faces = []
                for idx, angle in faces:
                    if not selected_faces or all(
                            abs(angle - prev_angle) > angle_similarity_threshold for _, prev_angle in selected_faces):
                        selected_faces.append((idx, angle))
                    else:
                        removed_indices.append(idx)

                # Store filtered data
                for idx, angle in selected_faces:
                    final_boxes[idx] = filtered_boxes[idx]
                    final_angles[idx] = angle

            # Update progress **after processing each group (person)**
            pbar.update(1)

    print(f"Removed {len(removed_indices)} redundant angles.")

    return final_boxes, final_angles, removed_indices


def create_2_channel_tensor(face_tensor, landmarks, point_size=2):
    """
    Args:
        face_tensor (torch.Tensor): Resized grayscale face tensor of shape (1, H, W).
        landmarks (list): Adjusted (x, y) landmark coordinates.
        point_size (int): Size of each landmark point in the heatmap.

    Returns:
        torch.Tensor: 2-channel tensor (face + landmark heatmap).
    """
    _, H, W = face_tensor.shape  # Should be (1, 512, 512)

    landmark_heatmap = torch.zeros((H, W), dtype=torch.float32)

    for x, y in landmarks:
        if 0 <= x < W and 0 <= y < H:
            # Set a small area around each landmark to 1
            landmark_heatmap[max(0, y-point_size):min(H, y+point_size+1),
                             max(0, x-point_size):min(W, x+point_size+1)] = 1.0

    final_tensor = torch.stack([face_tensor.squeeze(0), landmark_heatmap], dim=0)

    return final_tensor


def tensors_to_2_channels(loaded_data, point_size=2, num_workers=8):

    def process_single(idx, tensor, landmarks):
        """Applies 2-channel transformation to a single image."""
        return idx, create_2_channel_tensor(tensor, landmarks, point_size)

    all_entries = list(zip(loaded_data["idxs"], loaded_data["tensors"], loaded_data["landmarks"]))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda entry: process_single(*entry), all_entries),
            total=len(all_entries), desc="Creating 2-Channel Tensors"
        ))

    updated_data = {
        "idxs": [idx for idx, _ in results],
        "tensors": [tensor for _, tensor in results],
        "labels": loaded_data["labels"],  # Labels remain unchanged
    }

    return updated_data


class TwoChannelDataset(Dataset):
    def __init__(self, data):
        """
        Custom dataset for 2-channel tensors.
        Args:
            data (dict): Data containing "idxs", "tensors", "labels".
        """
        self.tensors = data["tensors"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        """
        Returns a single data sample (tensor, label).

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (tensor, label)
        """
        return self.tensors[idx], self.labels[idx]
