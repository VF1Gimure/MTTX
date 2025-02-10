import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os

# Fix: Import Weights
from torchvision.models import VGG16_Weights

# Use Metal (MPS) if available on Mac
device = torch.device("mps")

def train_model():
    # Load VGG16 with correct weights
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # Modify classifier for 7 emotion classes
    vgg16.classifier[6] = nn.Linear(4096, 7)
    vgg16 = vgg16.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    fer_dir = "/data/_expressions/Oheix"
    train_dir = fer_dir + "/train"
    val_dir = fer_dir + "/validation"

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Fix: Use num_workers = 0 on MacOS to avoid multiprocessing issues
    num_workers = 0  # MacOS issue fix
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training Phase
        vgg16.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(train_loader), acc=100 * correct / total)

        train_acc = 100 * correct / total

        # Validation Phase
        vgg16.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            loop = tqdm(val_loader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                outputs = vgg16(images)
                loss = criterion(outputs, labels)

                # Calculate validation accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
                loop.set_postfix(loss=val_loss / len(val_loader), acc=100 * val_correct / val_total)

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    print("Training complete.")

    # Save trained model
    torch.save(vgg16.state_dict(), "expression_vgg16.pth")

    return vgg16, transform

def predict_expression(image_path, model, transform):
    model.eval()  # Fix: Set model to evaluation mode
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    emotions = ["Angry", "Happy", "Neutral", "Fear", "Disgust", "Surprise", "Sad"]
    return emotions[predicted.item()]

import torch.multiprocessing as mp

if __name__ == '__main__':  # Fix: Prevent multiprocessing issue
    mp.set_start_method('spawn', force=True)  # Fix multiprocessing on Mac
    model, transform = train_model()

    # Fix: Reload model before prediction
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, 7)
    model.load_state_dict(torch.load("expression_vgg16.pth", map_location=device))
    model.to(device)
    model.eval()

    # Test example
    print(predict_expression("/data/_expressions/Mttx_clean/angry/angry_3.jpg", model, transform))
