import shutil
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DataframeTorchImageFolder:
    def __init__(self, dataset_dir, data_transforms=None):
        self.dataset_dir = dataset_dir
        self.save_dir = os.getcwd()
        self.data_transforms = data_transforms
        self.imagefolder_dataset = self._load_dataset()
        self.class_to_idx = self.imagefolder_dataset.class_to_idx
        self.df = self._create_dataframe()
        self.deleted_images = pd.DataFrame(columns=self.df.columns)

    def _load_dataset(self):
        return datasets.ImageFolder(self.dataset_dir, transform=self.data_transforms)

    def _create_dataframe(self):
        data = []
        for idx, (path, label) in enumerate(self.imagefolder_dataset.samples):
            data.append({"index": idx, "img_path": path, "class": label})
        return pd.DataFrame(data)

    def load_keep_transform(self, data_transforms=None):
        self.data_transforms = data_transforms
        self.imagefolder_dataset = self._load_dataset()
        valid_file_paths = set(self.df["img_path"])
        self.imagefolder_dataset.samples = [sample for sample in self.imagefolder_dataset.samples if
                                            sample[0] in valid_file_paths]
        self.imagefolder_dataset.targets = [label for _, label in self.imagefolder_dataset.samples]

    def pop(self, index):
        if index not in self.df.index:
            print(f"Index {index} not found in the current dataset.")
            return
        popped_row = self.df.loc[index]
        self.deleted_images = pd.concat([self.deleted_images, popped_row.to_frame().T], ignore_index=True)
        self.df = self.df.drop(index).reset_index(drop=True)

        file_path_to_remove = popped_row["img_path"]
        self.imagefolder_dataset.samples = [
            sample for sample
            in self.imagefolder_dataset.samples
            if sample[0] != file_path_to_remove]

        self.imagefolder_dataset.targets = [
            label for _, label
            in self.imagefolder_dataset.samples]

    def batch_pop(self, indices):
        if isinstance(indices, int):
            indices = [indices]

        invalid_indices = [idx for idx in indices if idx not in self.df.index]
        if invalid_indices:
            print(f"Indices {invalid_indices} not found in the current dataset.")
            indices = [idx for idx in indices if idx in self.df.index]

        popped_rows = self.df.loc[indices]
        self.deleted_images = pd.concat([self.deleted_images, popped_rows], ignore_index=True)
        self.df = self.df.drop(indices).reset_index(drop=True)

        valid_file_paths = set(self.df["img_path"])
        self.imagefolder_dataset.samples = [
            sample for sample in self.imagefolder_dataset.samples if sample[0] in valid_file_paths
        ]
        self.imagefolder_dataset.targets = [label for _, label in self.imagefolder_dataset.samples]

    def save(self, name):
        if not self.save_dir:
            raise ValueError("save_dir is not set.")
        os.makedirs(self.save_dir, exist_ok=True)

        df_path = os.path.join(self.save_dir, f"{name}_metadata.csv")
        self.df.to_csv(df_path, index=False)

        deleted_path = os.path.join(self.save_dir, f"{name}_deleted.csv")
        self.deleted_images.to_csv(deleted_path, index=False)

        if self.data_transforms is not None:
            transforms_path = os.path.join(self.save_dir, f"{name}_transforms.json")
            transforms_config = self._serialize_transforms(self.data_transforms)
            with open(transforms_path, "w") as f:
                json.dump(transforms_config, f)

    def clean_save(self, name):
        if not self.save_dir:
            raise ValueError("save_dir is not set.")
        os.makedirs(self.save_dir, exist_ok=True)

        # Directory to save the clean dataset
        clean_imagefolder_dir = os.path.join(self.save_dir, name)
        os.makedirs(clean_imagefolder_dir, exist_ok=True)

        # Remove existing `_deleted.csv` if it exists
        deleted_path = os.path.join(self.save_dir, f"{name}_deleted.csv")
        if os.path.exists(deleted_path):
            os.remove(deleted_path)

        # Copy valid files to the clean dataset directory
        for _, row in self.df.iterrows():
            file_path = row["img_path"]
            class_label = row["class"]
            class_dir = os.path.join(clean_imagefolder_dir, self.imagefolder_dataset.classes[class_label])
            os.makedirs(class_dir, exist_ok=True)

            dest_path = os.path.join(class_dir, os.path.basename(file_path))
            if not os.path.exists(dest_path):
                shutil.copy2(file_path, dest_path)

        # Save metadata
        df_path = os.path.join(self.save_dir, f"{name}_metadata.csv")
        self.df.to_csv(df_path, index=False)

    @classmethod
    def from_saved(cls, meta_dir, name, dataset_dir):
        """
        Alternative constructor to initialize the class from a saved state without running __init__.
        """
        df_path = os.path.join(meta_dir, f"{name}_metadata.csv")
        deleted_path = os.path.join(meta_dir, f"{name}_deleted.csv")
        transforms_path = os.path.join(meta_dir, f"{name}_transforms.json")

        if not os.path.exists(df_path):
            raise FileNotFoundError("The dataset metadata (DataFrame) file is missing.")
        if not os.path.exists(deleted_path):
            raise FileNotFoundError("The deleted images metadata is missing.")

        df = pd.read_csv(df_path)
        deleted_images = pd.read_csv(deleted_path)

        if os.path.exists(transforms_path):
            with open(transforms_path, "r") as f:
                data_transforms = cls._deserialize_transforms(json.load(f))
        else:
            data_transforms = None

        instance = cls.__new__(cls)
        instance.dataset_dir = dataset_dir
        instance.save_dir = meta_dir
        instance.data_transforms = data_transforms
        instance.df = df
        instance.deleted_images = deleted_images
        instance.imagefolder_dataset = None  # Needs to be reloaded using load_and_keep_state_new_transform
        instance.load_keep_transform(data_transforms)

        return instance

    @staticmethod
    def _serialize_transforms(data_transforms):
        serialized = []
        for t in data_transforms.transforms:
            transform_name = type(t).__name__
            params = t.__dict__
            serialized.append({"name": transform_name, "params": params})
        return serialized

    @staticmethod
    def _deserialize_transforms(serialized_transforms):
        transform_objects = []
        for t in serialized_transforms:
            transform_class = getattr(transforms, t["name"])
            transform_objects.append(transform_class(**t["params"]))
        return transforms.Compose(transform_objects)

