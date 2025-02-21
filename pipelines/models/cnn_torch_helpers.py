import torch
from tqdm import tqdm
import torch.nn.functional as F

def evaluate(model, data_loader, device='mps'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for xi, yi in data_loader:
            xi = xi.to(device, dtype=torch.float32)
            yi = yi.to(device, dtype=torch.long)

            scores = model(xi)
            _, predicted = torch.max(scores, 1)

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(yi.cpu().tolist())

    return predictions, true_labels


def accuracy(predictions, true_labels):
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    total = len(true_labels)
    return 100 * correct / total


def accuracy_direct(model, data_loader, device='mps'):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device=device)
    correct = 0
    total = 0

    with torch.inference_mode():  # No gradients required for evaluation
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


def train(model, optimizer, train_loader, epochs=10, device='mps', scheduler=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Trains the given model using the train loader data in the epochs specified.
    Prints the cost and accuracy of the model at each epoch.
    """

    model = model.to(device=device)
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for xi, yi in tepoch:
                xi = xi.to(device=device, dtype=torch.float32)
                yi = yi.to(device=device, dtype=torch.long)

                # Forward pass
                scores = model(xi)

                # Compute the cross-entropy loss
                cost = F.cross_entropy(input=scores, target=yi)

                # Zero gradients, backward pass, and optimization
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                #  running loss
                running_loss += cost.item()

                # Update progress bar
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        #  epoch info
        #print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')
        if scheduler is not None:
            scheduler.step()

    return model, epochs




def train(model, optimizer, train_loader, epochs=10, device='mps', scheduler=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Trains the given model using the train loader data in the epochs specified.
    Prints the cost and accuracy of the model at each epoch.
    """

    model = model.to(device=device)
    model.train()

    for epoch in range(epochs):

        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for xi, yi in tepoch:
                xi = xi.to(device=device, dtype=torch.float32)
                yi = yi.to(device=device, dtype=torch.long)

                # Forward pass
                scores = model(xi)

                # Compute the cross-entropy loss
                cost = F.cross_entropy(input=scores, target=yi)

                # Zero gradients, backward pass, and optimization
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                #  running loss
                running_loss += cost.item()

                # Update progress bar
                tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
                tepoch.set_postfix(loss=running_loss / len(train_loader))

        #  epoch info
        #print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}')
        if scheduler is not None:
            scheduler.step()

    return model, epochs


def train_by_ep(model, optimizer, train_loader, epochs=10, device='mps', scheduler=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device=device)
    model.train()

    with tqdm(range(epochs), unit="epoch") as pbar:
        for epoch in pbar:
            running_loss = 0.0

            for xi, yi in train_loader:
                xi = xi.to(device=device, dtype=torch.float32)
                yi = yi.to(device=device, dtype=torch.long)

                # Forward pass
                scores = model(xi)

                # Compute the cross-entropy loss
                cost = F.cross_entropy(input=scores, target=yi)

                # Zero gradients, backward pass, and optimization
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Running loss
                running_loss += cost.item()

            avg_loss = running_loss / len(train_loader)
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            pbar.set_postfix(loss=avg_loss)

            if scheduler is not None:
                scheduler.step()

    return model, epochs

