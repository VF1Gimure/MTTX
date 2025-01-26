import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from mttx_cnn import MTTX_CNN
from mttx_seq import MTTX_Seq
from cnn_custom import CNN_custom
from cnn_3 import CNN3, ConvBlock
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from torchvision import datasets, transforms
from tqdm import tqdm  # Import tqdm for progress bars
from torchvision import models


data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

def load_data(train_dir, test_dir, batch_size=32):
    """
    Loads the training and testing datasets and returns data loaders.
    """
    # Load datasets using the common transformation
    train_data = datasets.ImageFolder(train_dir, data_transforms)
    test_data = datasets.ImageFolder(test_dir, data_transforms)

    # Create DataLoader instances for training and testing datasets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_Net_data(data_dir, batch, ratio):
    full_data = datasets.ImageFolder(data_dir, data_transforms)

    total_size = len(full_data)
    train_size = int(total_size * ratio)
    test_size = total_size - train_size

    train_data, test_data = random_split(full_data, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)

    return train_loader, test_loader


def accuracy(model, data_loader, device='cpu'):
    """
    Calculate the accuracy of the model on the given dataset (validation set).
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients required for evaluation
        for xi, yi in data_loader:
            xi = xi.to(device=device, dtype=torch.float32)
            yi = yi.to(device=device, dtype=torch.long)

            # Get model predictions
            scores = model(xi)
            _, predicted = torch.max(scores, 1)

            # Update correct and total counts
            total += yi.size(0)
            correct += (predicted == yi).sum().item()

    accuracy = 100 * correct / total
    return accuracy


import os
from tqdm import tqdm
import torch.nn.functional as F
import torch


def train(model, optimizer, train_loader, criterion, val_loader, epochs=10, device='cpu', save_path='../model/models/', model_name='best_model.pth'):

    model = model.to(device)  # Move model to the specified device
    best_val_acc = 0.0  # Track the best validation accuracy
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Accumulate training loss
        correct_train = 0
        total_train = 0

        # Training loop with progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            for xi, yi in tepoch:
                xi, yi = xi.to(device, dtype=torch.float32), yi.to(device, dtype=torch.long)

                # Forward pass
                scores = model(xi)

                # Compute loss
                loss = criterion(scores, yi)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                running_loss += loss.item()
                _, preds = torch.max(scores, 1)
                correct_train += (preds == yi).sum().item()
                total_train += yi.size(0)

                # Update progress bar
                tepoch.set_postfix(
                    loss=running_loss / len(train_loader),
                    acc=(correct_train / total_train) * 100
                )

        # Calculate training accuracy
        train_acc = (correct_train / total_train) * 100
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate on validation set
        model.eval()  # Set model to evaluation mode
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        with torch.no_grad():
            for xi, yi in val_loader:
                xi, yi = xi.to(device, dtype=torch.float32), yi.to(device, dtype=torch.long)

                scores = model(xi)
                val_loss += criterion(scores, yi).item()
                _, preds = torch.max(scores, 1)
                correct_val += (preds == yi).sum().item()
                total_val += yi.size(0)

        val_acc = (correct_val / total_val) * 100
        val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_path, model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Val Acc: {best_val_acc:.2f}% at {best_model_path}")


# Main script to run training
if __name__ == '__main__':
    train_dir = '../data/FER/train'
    test_dir = '../data/FER/test'
    NET_dir = '../data/NETv3'
    # Load the datasets
    # train_loader, test_loader = load_data(train_dir, test_dir)
    train_loader, test_loader = load_Net_data(NET_dir, 32, .8)
    resnet = models.resnet18(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    num_classes = 8  # For expression classification
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet.to('cpu')

    optimizer = optim.Adam(resnet.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    #train(resnet, train_loader, criterion, optimizer, device, epochs=10)
    train(
        model=resnet,
        optimizer=optimizer,
        train_loader=train_loader,
        criterion=criterion,
        val_loader=test_loader,
        epochs=10,
        device='cpu',
        save_path='../model/models/',
        model_name='resnet18.pth'
    )
    
    final_accuracy = accuracy(resnet, test_loader, device='cpu')
    print(f'Final Accuracy on Test Set: {final_accuracy:.2f}%')