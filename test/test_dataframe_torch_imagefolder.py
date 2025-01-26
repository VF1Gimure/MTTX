import pytest
import os
import shutil
import pandas as pd
from torchvision import transforms
from utils.dataframe_torch_imagefolder import DataframeTorchImageFolder
from PIL import Image

@pytest.fixture(scope="module")
def setup_test_environment():
    # Setup: Create test dataset directory
    test_dataset_dir = "test_dataset"
    test_save_dir = "test_save"
    os.makedirs(test_dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dataset_dir, "class1"), exist_ok=True)
    os.makedirs(os.path.join(test_dataset_dir, "class2"), exist_ok=True)

    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(os.path.join(test_dataset_dir, "class1", f"image1_{i}.jpg"))

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(0, 0, 0))
        img.save(os.path.join(test_dataset_dir, "class2", f"image2_{i}.jpg"))

    yield test_dataset_dir, test_save_dir

    # Teardown: Remove test directories
    shutil.rmtree(test_dataset_dir)
    shutil.rmtree(test_save_dir, ignore_errors=True)


def test_initialization(setup_test_environment):
    test_dataset_dir, _ = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir)

    assert len(loader.df) == 8
    assert loader.df["class_label"].nunique() == 2
    assert loader.deleted_images.empty


def test_pop_single_image(setup_test_environment):
    test_dataset_dir, _ = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir)

    loader.pop(0)
    assert len(loader.df) == 7
    assert len(loader.deleted_images) == 1
    assert loader.deleted_images.iloc[0]["index"] == 0


def test_batch_pop(setup_test_environment):
    test_dataset_dir, _ = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir)

    loader.batch_pop([0, 1, 2])
    assert len(loader.df) == 5
    assert len(loader.deleted_images) == 3


def test_save_and_reload(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.pop(0)
    loader.save_dir = test_save_dir
    loader.save(name="test_checkpoint")

    # Check saved files
    assert os.path.exists(os.path.join(test_save_dir, "test_checkpoint_metadata.csv"))
    assert os.path.exists(os.path.join(test_save_dir, "test_checkpoint_deleted.csv"))

    # Reload
    reloaded_loader = DataframeTorchImageFolder.from_saved(
        meta_dir=test_save_dir,
        name="test_checkpoint",
        dataset_dir=test_dataset_dir,
    )

    assert len(reloaded_loader.df) == 7
    assert len(reloaded_loader.deleted_images) == 1


def test_clean_save(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.save_dir = test_save_dir

    loader.pop(0)
    loader.pop(1)

    clean_name = "clean_dataset"
    loader.clean_save(name=clean_name)

    # Check that the clean save directory exists
    clean_imagefolder_dir = os.path.join(test_save_dir, clean_name)
    assert os.path.exists(clean_imagefolder_dir), f"Clean save directory not found: {clean_imagefolder_dir}"

    # Check that deleted files are not in the clean save
    deleted_paths = loader.deleted_images["file_path"].tolist()
    for deleted_path in deleted_paths:
        deleted_file_name = os.path.basename(deleted_path)
        for root, _, files in os.walk(clean_imagefolder_dir):
            assert deleted_file_name not in files, f"Deleted file {deleted_file_name} found in clean save!"

    # Check that valid files are copied to the clean save
    valid_paths = loader.df["file_path"].tolist()
    for valid_path in valid_paths:
        valid_file_name = os.path.basename(valid_path)
        found = False
        for root, _, files in os.walk(clean_imagefolder_dir):
            if valid_file_name in files:
                found = True
                break
        assert found, f"Valid file {valid_file_name} not found in clean save!"

    # Check the metadata file
    metadata_path = os.path.join(test_save_dir, f"{clean_name}_metadata.csv")
    assert os.path.exists(metadata_path), f"Metadata file not found: {metadata_path}"

    # Check that deleted metadata file is removed
    deleted_metadata_path = os.path.join(test_save_dir, f"{clean_name}_deleted.csv")
    assert not os.path.exists(
        deleted_metadata_path), f"Deleted metadata file should not exist: {deleted_metadata_path}"


def test_batch_pop_single_index(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.save_dir = test_save_dir

    # Perform batch_pop with a single index
    loader.batch_pop(0)

    # Assertions
    assert len(loader.df) == 7, "Dataframe size should decrease by 1 after batch_pop with a single index."
    assert len(loader.deleted_images) == 1, "Deleted images size should increase by 1 after batch_pop with a single index."
    assert loader.deleted_images.iloc[0]["index"] == 0, "The popped index should match the deleted image index."

def test_batch_pop_multiple_operations(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.save_dir = test_save_dir

    # Track file paths for validation
    initial_file_paths = loader.df["file_path"].tolist()
    first_batch_files = initial_file_paths[:2]
    second_batch_files = initial_file_paths[2:4]

    # Perform the first batch_pop operation (remove first two images)
    loader.batch_pop([0, 1])  # Indices before re-indexing

    # Assertions after the first operation
    remaining_files = loader.df["file_path"].tolist()
    deleted_files = loader.deleted_images["file_path"].tolist()
    assert all(f not in remaining_files for f in first_batch_files), "First batch files should be removed from the dataset."
    assert all(f in deleted_files for f in first_batch_files), "First batch files should be in the deleted images."

    # Perform the second batch_pop operation (remove next two images after re-indexing)
    loader.batch_pop([0, 1])  # Adjusted indices due to re-indexing

    # Assertions after the second operation
    remaining_files = loader.df["file_path"].tolist()
    deleted_files = loader.deleted_images["file_path"].tolist()
    assert all(f not in remaining_files for f in second_batch_files), "Second batch files should be removed from the dataset."
    assert all(f in deleted_files for f in second_batch_files), "Second batch files should be in the deleted images."


def test_batch_pop_invalid_indices(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.save_dir = test_save_dir
    # Perform batch_pop with some valid and some invalid indices
    loader.batch_pop([0, 100])  # Index 100 is invalid

    # Assertions
    assert len(loader.df) == 7, "Dataframe size should decrease by 1 after batch_pop with one valid index."
    assert len(loader.deleted_images) == 1, "Deleted images size should increase by 1 after batch_pop with one valid index."
    assert 0 in loader.deleted_images["index"].values, "Valid index should be deleted."
    assert 100 not in loader.deleted_images["index"].values, "Invalid index should not be deleted."


def test_batch_pop_no_indices(setup_test_environment):
    test_dataset_dir, test_save_dir = setup_test_environment
    loader = DataframeTorchImageFolder(dataset_dir=test_dataset_dir, data_transforms=None)

    loader.save_dir = test_save_dir
    # Perform batch_pop with an empty list
    loader.batch_pop([])

    # Assertions
    assert len(loader.df) == 8, "Dataframe size should not change after batch_pop with no indices."
    assert len(loader.deleted_images) == 0, "Deleted images size should not change after batch_pop with no indices."
