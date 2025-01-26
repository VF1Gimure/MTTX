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

# Common data transformation
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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


def train(model, optimizer, train_loader, val_loader, epochs=10, device='cpu', save='../model/models/'):
    """
    Trains the given model using the train loader data in the epochs specified.
    Prints the cost and accuracy of the model at each epoch.
    """
    model = model.to(device=device)  # Move model to the specified device (CPU/GPU)

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0  # Initialize loss accumulator

        # Using tqdm to show progress in each epoch
        with tqdm(train_loader, unit="batch") as tepoch:
            for xi, yi in tepoch:
                xi = xi.to(device=device, dtype=torch.float32)  # Input data to device
                yi = yi.to(device=device, dtype=torch.long)  # Target labels to device

                # Forward pass
                scores = model(xi)

                # Compute the cross-entropy loss
                cost = F.cross_entropy(input=scores, target=yi)

                # Zero gradients, backward pass, and optimization
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Accumulate running loss
                running_loss += cost.item()

                # Update progress bar description
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        # Print epoch information
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')

        # Calculate accuracy on validation set after each epoch
        acc = accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {acc:.2f}%')

    # Save the model after training is complete
    torch.save(model.state_dict(), save)
    print('Model saved!')


# Main script to run training
if __name__ == '__main__':
    train_dir = '../data/FER/train'
    test_dir = '../data/FER/test'
    NET_dir = '../data/NETv3'
    # Load the datasets
    # train_loader, test_loader = load_data(train_dir, test_dir)
    train_loader, test_loader = load_Net_data(NET_dir, 32, .8)
    # Initialize model, optimizer
    # _model = MTTX_CNN(in_channel=1, channel1=32, channel2=64, channel3=128, channel4=256)
    conv_k3 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    channels = [32, 64, 128, 256]
    out_features = 8
    epochs = 10
    lr = 1e-4
    cnn_image_net = nn.Sequential(  # We can construct the new model by adding our modules here
        CNN_custom(3, channels[0], channels[0]),
        CNN_custom(channels[0], channels[1], channels[1]),
        CNN_custom(channels[1], channels[2], channels[2]),
        CNN_custom(channels[2], channels[3], channels[3]),
        nn.Flatten(),
        nn.Linear(in_features=4 * 4 * channels[3], out_features=out_features))  # 48/2/2/2/2 = 3
    optimiser = torch.optim.Adam(cnn_image_net.parameters(), lr=lr)

    # Train
    train(cnn_image_net, optimiser, train_loader, test_loader, epochs=10, device='cpu', save='../model/models/cnn_image_net_resnet.pth')

    final_accuracy = accuracy(cnn_image_net, test_loader, device='cpu')
    print(f'Final Accuracy on Test Set: {final_accuracy:.2f}%')