import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.data_augmentation import augment_data

# load the dataset
print("Loading dataset...")
data = pd.read_csv("data/expert_policy_data.csv")

# convert board states to input features
unique_data = augment_data(data)

# Convert to arrays
X_final = np.array([np.array(key).reshape(4, 4)
                   for key in unique_data.keys()], dtype=np.float32)
y_final = np.array(list(unique_data.values()), dtype=np.int64)

# reshape for CNN
X_final = X_final.reshape(-1, 1, 4, 4)

# split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y_final, test_size=0.1, random_state=42)

# convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"Size of training dataset: {len(train_dataset)}")
# define the policy network


class PolicyNetworkCNN(nn.Module):
    def __init__(self):
        super(PolicyNetworkCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = PolicyNetworkCNN()
class_counts = np.bincount(y_train)
class_weights = [len(y_train) / count for count in class_counts]

# convert class_weights to float32
class_weights = np.array(class_weights, dtype=np.float32)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training loop


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
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
        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = 100 * correct / total
        print(
            f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")


# train the model
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100)

# save the trained model
torch.save(model.state_dict(), "data/policy_network.pth")
