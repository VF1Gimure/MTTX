import os

from PIL import Image
from matplotlib import pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D

def plot_random_images(data_dir, images_df, title, sample=5):
    # Select 10 random images
    random_samples = images_df.sample(n=sample)

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    fig.suptitle(title, fontsize=15)

    for ax, (_, row) in zip(axes.flatten(), random_samples.iterrows()):
        img_path = row["img_path"]
        label = row["class"]
        # class_name = dataset.classes[label]

        try:
            img = Image.open(img_path)
            # print(img_path)
            name = os.path.basename(img_path.replace(data_dir, ""))
            parent_folder = os.path.dirname(img_path.replace(data_dir, "")).split(os.sep)[-1]
            ax.imshow(img)
            ax.set_title(str(label) + " " + name + " " + parent_folder, fontsize=12)
            ax.axis('off')
        except Exception as e:
            ax.set_title("Error loading image")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_random_images_dfolder(dataset, title, sample=5):
    """
    Plots a random selection of images from a PyTorch ImageFolder dataset.

    :param dataset: A dataset object created using torchvision.datasets.ImageFolder.
    :param title: Title of the plot.
    :param sample: Number of random images to display.
    """
    # Select `sample` random indices from the dataset
    indices = random.sample(range(len(dataset)), sample)

    fig, axes = plt.subplots(1, sample, figsize=(sample * 3, 5))
    fig.suptitle(title, fontsize=15)

    for ax, idx in zip(axes.flatten(), indices):
        img_path, label = dataset.samples[idx]  # Get image path and label index
        class_name = dataset.classes[label]  # Convert label index to class name

        try:
            img = Image.open(img_path)
            name = os.path.basename(img_path)  # Image filename
            parent_folder = os.path.basename(os.path.dirname(img_path))  # Class folder

            ax.imshow(img)
            ax.set_title(f"{class_name} ({label})\n{name} | {parent_folder}", fontsize=10)
            ax.axis('off')
        except Exception as e:
            ax.set_title("Error loading image")
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_image(image_tensor):
    """
    Display an image tensor in Jupyter Notebook.
    """
    # If normalized, unnormalize it (adjust mean and std as per your transforms)
    mean = [0.5]
    std = [0.5]
    image_tensor = image_tensor * std[0] + mean[0]

    # Convert to NumPy array and move channels last
    img = image_tensor.permute(1, 2, 0).numpy()

    # If single channel, remove the channel dimension
    if img.shape[-1] == 1:
        img = img.squeeze()

    # Plot using Matplotlib
    plt.imshow(img,cmap="gray")
    plt.axis("off")  # Turn off axes
    plt.show()


def show_images(image_tensor1, image_tensor2, title1="Image 1", title2="Image 2"):
    """
    Display two image tensors side by side in Jupyter Notebook with titles.
    """
    mean = [0.5]
    std = [0.5]

    # Unnormalize both images
    image_tensor1 = image_tensor1 * std[0] + mean[0]
    image_tensor2 = image_tensor2 * std[0] + mean[0]

    # Convert to NumPy arrays and move channels last
    img1 = image_tensor1.permute(1, 2, 0).numpy()
    img2 = image_tensor2.permute(1, 2, 0).numpy()

    # If single channel, remove the channel dimension
    if img1.shape[-1] == 1:
        img1 = img1.squeeze()
    if img2.shape[-1] == 1:
        img2 = img2.squeeze()

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(30, 15))

    axes[0].imshow(img1, cmap="gray")
    axes[0].set_title(title1, fontsize=40, fontweight="bold")
    axes[0].axis("off")  # Turn off axes

    axes[1].imshow(img2,cmap="gray")
    axes[1].set_title(title2, fontsize=40, fontweight="bold")
    axes[1].axis("off")  # Turn off axes

    plt.show()


def plot_landmarks_3d(landmarks_3d, title="Landmarks 3D"):
    """
    Visualiza los landmarks faciales en un gráfico 3D usando Matplotlib.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extraer coordenadas X, Y, Z
    x = landmarks_3d[:, 0]
    y = landmarks_3d[:, 1]
    z = landmarks_3d[:, 2]

    ax.scatter(x, y, z, c='r', marker='o', s=10)  # Puntos rojos para los landmarks

    # Etiquetas de los ejes
    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.set_zlabel("Eje Z")

    ax.set_title(title)
    plt.show()

# Suponiendo que landmarks_3d es el resultado de transform_landmarks_to_3d()
# landmarks_3d = transform_landmarks_to_3d(landmarks_2d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
# plot_landmarks_3d(landmarks_3d
