import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.data_augmentation import augment_data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import os


def load_dataset(file_path):
    """
    Load and preprocess the dataset.
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    unique_data = augment_data(data)

    # Convert board states to input features
    X = np.array([np.array(key).reshape(4, 4)
                  for key in unique_data.keys()], dtype=np.float32)
    y = np.array(list(unique_data.values()), dtype=np.int64)

    # Reshape for CNN
    X = X.reshape(-1, 1, 4, 4)
    return X, y


def prepare_dataloaders(X, y, batch_size=64, test_size=0.1):
    """
    Split the data into training and validation sets and create dataloaders.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Size of training dataset: {len(train_dataset)}")
    print(f"Size of validation dataset: {len(val_dataset)}")

    return train_loader, val_loader, y_train


class PolicyNetworkCNN(nn.Module):
    def __init__(self):
        super(PolicyNetworkCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def compute_class_weights(y_train):
    """
    Compute class weights to handle class imbalance.
    """
    class_counts = np.bincount(y_train)
    class_weights = [len(y_train) / count for count in class_counts]
    class_weights = np.array(class_weights, dtype=np.float32)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50, patience=5):
    """
    Train the model with optional early stopping.
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            # Move data to the same device as the model
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_accuracy = 100 * correct / total
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Move data to the same device as the model
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "data/policy_network_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def main():
    # Set file paths
    data_file_path = "data/expert_policy_data_early.csv"
    model_save_path = "data/policy_network.pth"

    # Load and preprocess the dataset
    X, y = load_dataset(data_file_path)

    # Prepare dataloaders
    train_loader, val_loader, y_train = prepare_dataloaders(X, y)

    device = torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print(f"Using device: {device}")

    # Initialize the model
    model = PolicyNetworkCNN().to(device)

    # Compute class weights for handling class imbalance
    class_weights = compute_class_weights(y_train).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    train_model(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device, epochs=20, patience=5)

    # Load the best model
    model.load_state_dict(torch.load("data/policy_network_best.pth"))

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
