import os

from PIL import Image
from matplotlib import pyplot as plt
import random

from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def visualize_2_channel_image(data, label_map, idx=0):
    """
    Visualizes a 2-channel tensor image (grayscale + landmark heatmap)
    and prints the category label.

    Args:
        data (dict): Dataset containing idxs, tensors, and labels.
        label_map (dict): Dictionary mapping labels to categories.
        idx (int): Index of the image to visualize.
    """
    tensor = data["tensors"][idx]
    label = data["labels"][idx]

    # Extract the grayscale face (channel 0) and landmark heatmap (channel 1)
    face = tensor[0].cpu().numpy()
    heatmap = tensor[1].cpu().numpy()

    # Display grayscale face
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(face, cmap="gray")
    plt.title(f"Face - Category: {label_map[label]}")
    plt.axis("off")

    # Overlay heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(face, cmap="gray")
    plt.imshow(heatmap, cmap="hot", alpha=0.6)  # Blend heatmap with the face
    plt.title("Landmark Heatmap Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_all_class_cm(cm, class_labels=None, per_row=3):
    num_classes = cm.shape[0]

    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(num_classes)]

    # Define grid layout
    cols = per_row
    rows = math.ceil(num_classes / cols)  # Automatically determine number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size
    axes = axes.flatten()  # Flatten in case of uneven grids

    for class_idx in range(num_classes):
        ax = axes[class_idx]

        # Normalize confusion matrix rows to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        # Create cell labels (only for TP, FN, FP)
        cell_labels = np.full((num_classes, num_classes), "", dtype=object)  # Empty by default
        cell_colors = np.empty((num_classes, num_classes), dtype=object)

        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                percent = cm_percent[i, j]
                if i == class_idx and j == class_idx:  # True Positives
                    cell_labels[i, j] = f"TP\n{count}\n{percent:.1f}%"
                    color = "lightblue"
                elif i == class_idx:  # False Negatives
                    cell_labels[i, j] = f"FN\n{count}\n{percent:.1f}%"
                    color = "lightgreen"
                elif j == class_idx:  # False Positives
                    cell_labels[i, j] = f"FP\n{count}\n{percent:.1f}%"
                    color = "lightcoral"
                else:  # True Negatives
                    cell_labels[i, j] = f"TN\n{count}"
                    color = "beige"

                cell_colors[i, j] = color

        # Plot the confusion matrix
        sns.heatmap(
            cm_percent, annot=cell_labels, fmt='', cmap="Pastel1",
            xticklabels=class_labels, yticklabels=class_labels,
            cbar=False, ax=ax
        )

        # Apply custom cell colors
        for i in range(num_classes):
            for j in range(num_classes):
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, facecolor=cell_colors[i, j], edgecolor='black', lw=0.5
                ))

        # Add bounding boxes for the highlighted class
        for i in range(num_classes):
            for j in range(num_classes):
                if i == class_idx and j == class_idx:  # True Positive
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=1.5, linestyle='dashed'))
                elif i == class_idx:  # False Negative
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=1, linestyle='dashed'))
                elif j == class_idx:  # False Positive
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1, linestyle='dashed'))

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"CM: {class_labels[class_idx]}")

    # Hide any unused subplots
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def highlight_class_cm(cm, class_idx, class_labels=None):
    num_classes = cm.shape[0]

    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(num_classes)]

    # Normalize confusion matrix rows to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    # Create cell labels and define colors
    cell_labels = []
    cell_colors = np.empty((num_classes, num_classes), dtype=object)
    for i in range(num_classes):
        row_labels = []
        for j in range(num_classes):
            count = cm[i, j]
            percent = cm_percent[i, j]
            if i == class_idx and j == class_idx:  # True Positives
                label = f"TP\n{count}\n{percent:.1f}%"
                color = "lightblue"
            elif i == class_idx:  # False Negatives
                label = f"FN\n{count}\n{percent:.1f}%"
                color = "lightgreen"
            elif j == class_idx:  # False Positives
                label = f"FP\n{count}\n{percent:.1f}%"
                color = "lightcoral"
            else:  # True Negatives
                label = f"TN\n{count}\n{percent:.1f}%"
                color = "beige"

            row_labels.append(label)
            cell_colors[i, j] = color
        cell_labels.append(row_labels)

    # Plot the confusion matrix with heatmap
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        cm_percent,
        annot=np.array(cell_labels),
        fmt='',
        cmap="Pastel1",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False
    )

    # Apply custom cell colors
    for i in range(num_classes):
        for j in range(num_classes):
            ax.add_patch(plt.Rectangle(
                (j, i), 1, 1, facecolor=cell_colors[i, j], edgecolor='black', lw=0.5
            ))

    # Add bounding boxes for the highlighted class
    for i in range(num_classes):
        for j in range(num_classes):
            if i == class_idx and j == class_idx:  # True Positive
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=1.5, linestyle='dashed'))
            elif i == class_idx:  # False Negative
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=1, linestyle='dashed'))
            elif j == class_idx:  # False Positive
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1, linestyle='dashed'))

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix Highlight for Class {class_labels[class_idx]}")
    plt.show()


import math


def plot_all_class_cm(cm, class_labels=None, per_row=3):
    num_classes = cm.shape[0]

    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(num_classes)]

    # Define grid layout
    cols = per_row
    rows = math.ceil(num_classes / cols)  # Automatically determine number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size
    axes = axes.flatten()  # Flatten in case of uneven grids

    for class_idx in range(num_classes):
        ax = axes[class_idx]

        # Normalize confusion matrix rows to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

        # Create cell labels (only for TP, FN, FP)
        cell_labels = np.full((num_classes, num_classes), "", dtype=object)  # Empty by default
        cell_colors = np.empty((num_classes, num_classes), dtype=object)

        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                percent = cm_percent[i, j]
                if i == class_idx and j == class_idx:  # True Positives
                    cell_labels[i, j] = f"TP\n{count}\n{percent:.1f}%"
                    color = "lightblue"
                elif i == class_idx:  # False Negatives
                    cell_labels[i, j] = f"FN\n{count}\n{percent:.1f}%"
                    color = "lightgreen"
                elif j == class_idx:  # False Positives
                    cell_labels[i, j] = f"FP\n{count}\n{percent:.1f}%"
                    color = "lightcoral"
                else:  # True Negatives
                    cell_labels[i, j] = f"TN\n{count}"
                    color = "beige"

                cell_colors[i, j] = color

        # Plot the confusion matrix
        sns.heatmap(
            cm_percent, annot=cell_labels, fmt='', cmap="Pastel1",
            xticklabels=class_labels, yticklabels=class_labels,
            cbar=False, ax=ax
        )

        # Apply custom cell colors
        for i in range(num_classes):
            for j in range(num_classes):
                ax.add_patch(plt.Rectangle(
                    (j, i), 1, 1, facecolor=cell_colors[i, j], edgecolor='black', lw=0.5
                ))

        # Add bounding boxes for the highlighted class
        for i in range(num_classes):
            for j in range(num_classes):
                if i == class_idx and j == class_idx:  # True Positive
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=1.5, linestyle='dashed'))
                elif i == class_idx:  # False Negative
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='green', lw=1, linestyle='dashed'))
                elif j == class_idx:  # False Positive
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=1, linestyle='dashed'))

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"CM: {class_labels[class_idx]}")

    # Hide any unused subplots
    for i in range(num_classes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
