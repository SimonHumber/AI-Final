#!/usr/bin/env python3
"""
Payment Risk Classification using PyTorch Neural Network
Based on notebook-style implementation - Simplified Version
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as td
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
print("Loading dataset...")
data = pd.read_csv("risk-dataset.txt", sep="\t").dropna()

# Simple preprocessing - convert categorical to numerical
for col in data.columns:
    if data[col].dtype == "object" and col != "CLASS":
        data[col] = pd.Categorical(data[col]).codes

# Separate features and target
features = data.drop(["ORDER_ID", "CLASS"], axis=1).columns
X = data[features].values
y = data["CLASS"].map({"no": 0, "yes": 1}).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data 70%-30% like notebook
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Training Set: {len(X_train)}, Test Set: {len(X_test)}")

# Create datasets and loaders like notebook
train_x = torch.Tensor(X_train).float()
train_y = torch.Tensor(y_train).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=0)

test_x = torch.Tensor(X_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x, test_y)
test_loader = td.DataLoader(test_ds, batch_size=20, shuffle=False, num_workers=0)

print("Ready to load data")

# Define neural network like notebook
hl = 10  # Number of hidden layer nodes


class RiskNet(nn.Module):
    def __init__(self):
        super(RiskNet, self).__init__()
        self.fc1 = nn.Linear(len(features), hl)
        self.fc2 = nn.Linear(hl, hl)
        self.fc3 = nn.Linear(hl, 2)  # 2 classes for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


# Create model instance
model = RiskNet()
print(model)


# Training functions like notebook
def train(model, data_loader, optimizer):
    model.train()
    train_loss = 0

    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / (batch + 1)
    print(f"Training set: Average loss: {avg_loss:.6f}")
    return avg_loss


def test(model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            out = model(data)
            test_loss += loss_criteria(out, target).item()
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()

    avg_loss = test_loss / batch_count
    print(
        f"Validation set: Average loss: {avg_loss:.6f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)\n"
    )
    return avg_loss


# Specify loss criteria and optimizer like notebook
loss_criteria = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Track metrics like notebook
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 50 epochs like notebook
epochs = 50
for epoch in range(1, epochs + 1):
    print(f"Epoch: {epoch}")
    train_loss = train(model, train_loader, optimizer)
    test_loss = test(model, test_loader)

    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

# Plot training history like notebook
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["training", "validation"], loc="upper right")
plt.title("Training and Validation Loss")
plt.savefig("notebook_style_training_history.png", dpi=300, bbox_inches="tight")
plt.show()

# Evaluate model
model.eval()
with torch.no_grad():
    x = torch.Tensor(X_test).float()
    _, predicted = torch.max(model(x).data, 1)

# Calculate metrics
accuracy = (predicted.numpy() == y_test).mean()
print(f"Final Accuracy: {accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, predicted.numpy())
print("Confusion Matrix:")
print(cm)

# Save model like notebook
torch.save(model.state_dict(), "notebook_style_risk_model.pt")
print("Model saved as notebook_style_risk_model.pt")
