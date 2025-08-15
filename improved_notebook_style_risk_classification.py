#!/usr/bin/env python3
"""
Improved Payment Risk Classification using PyTorch Neural Network
Based on notebook-style implementation with class imbalance handling
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
    f1_score,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
print("Loading dataset...")
data = pd.read_csv("risk-dataset.txt", sep="\t")

# Handle missing values
data = data.fillna("unknown")

# Enhanced preprocessing - convert categorical to numerical with better encoding
print("Enhanced preprocessing...")

# Binary risk indicators
risk_indicators = [
    "B_EMAIL",
    "B_TELEFON",
    "FLAG_LRIDENTISCH",
    "FLAG_NEWSLETTER",
    "CHK_LADR",
    "CHK_RADR",
    "CHK_KTO",
    "CHK_CARD",
    "CHK_COOKIE",
    "CHK_IP",
    "FAIL_LPLZ",
    "FAIL_LORT",
    "FAIL_LPLZORTMATCH",
    "FAIL_RPLZ",
    "FAIL_RORT",
    "FAIL_RPLZORTMATCH",
]

for col in risk_indicators:
    if col in data.columns:
        data[col] = (data[col] == "yes").astype(int)

# Payment method encoding
if "Z_METHODE" in data.columns:
    data["Z_METHODE"] = (
        data["Z_METHODE"]
        .map({"credit_card": 1, "check": 2, "debit_note": 3, "?": 0})
        .fillna(0)
    )

# Card type encoding
if "Z_CARD_ART" in data.columns:
    data["Z_CARD_ART"] = (
        data["Z_CARD_ART"]
        .map({"Visa": 1, "Eurocard": 2, "Mastercard": 3, "?": 0})
        .fillna(0)
    )

# Weekday encoding
if "WEEKDAY_ORDER" in data.columns:
    weekday_map = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7,
    }
    data["WEEKDAY_ORDER"] = data["WEEKDAY_ORDER"].map(weekday_map).fillna(0)

# Customer type
if "NEUKUNDE" in data.columns:
    data["NEUKUNDE"] = (data["NEUKUNDE"] == "yes").astype(int)

# Convert numerical columns
numerical_cols = [
    "VALUE_ORDER",
    "AMOUNT_ORDER",
    "SESSION_TIME",
    "AMOUNT_ORDER_PRE",
    "VALUE_ORDER_PRE",
]
for col in numerical_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

# Create enhanced features
print("Creating enhanced features...")
data["RISK_FACTORS"] = data[
    ["CHK_LADR", "CHK_RADR", "CHK_KTO", "CHK_CARD", "CHK_COOKIE", "CHK_IP"]
].sum(axis=1)
data["ADDRESS_FAILURES"] = data[
    [
        "FAIL_LPLZ",
        "FAIL_LORT",
        "FAIL_LPLZORTMATCH",
        "FAIL_RPLZ",
        "FAIL_RORT",
        "FAIL_RPLZORTMATCH",
    ]
].sum(axis=1)
data["TOTAL_RISK_SCORE"] = data["RISK_FACTORS"] + data["ADDRESS_FAILURES"]
data["HIGH_VALUE_ORDER"] = (
    data["VALUE_ORDER"] > data["VALUE_ORDER"].quantile(0.9)
).astype(int)
data["LONG_SESSION"] = (
    data["SESSION_TIME"] > data["SESSION_TIME"].quantile(0.9)
).astype(int)
data["NEW_CUSTOMER_HIGH_VALUE"] = (data["NEUKUNDE"] == 1) & (
    data["HIGH_VALUE_ORDER"] == 1
)
data["NEW_CUSTOMER_HIGH_VALUE"] = data["NEW_CUSTOMER_HIGH_VALUE"].astype(int)

# Drop columns that were transformed
columns_to_drop = [
    "B_BIRTHDATE",
    "Z_CARD_VALID",
    "TIME_ORDER",
    "DATE_LORDER",
    "Z_LAST_NAME",
    "MAHN_AKT",
    "MAHN_HOECHST",
]
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Drop ANUMMER columns
anummer_cols = [col for col in data.columns if col.startswith("ANUMMER")]
data = data.drop(anummer_cols, axis=1)

# Separate features and target
features = data.drop(["ORDER_ID", "CLASS"], axis=1).columns
X = data[features].values
y = data["CLASS"].map({"no": 0, "yes": 1}).values

print(f"Original class distribution:")
print(f"Low risk (0): {np.sum(y == 0)}")
print(f"High risk (1): {np.sum(y == 1)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data 70%-30% like notebook
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Training Set: {len(X_train)}, Test Set: {len(X_test)}")

# Balance the training data
print("\nBalancing training data...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Apply undersampling if still imbalanced
if np.sum(y_train_balanced == 1) / len(y_train_balanced) < 0.3:
    rus = RandomUnderSampler(random_state=42, sampling_strategy=0.4)
    X_train_balanced, y_train_balanced = rus.fit_resample(
        X_train_balanced, y_train_balanced
    )

print(f"Balanced training set shape: {X_train_balanced.shape}")
print(f"Balanced class distribution:")
print(f"Low risk (0): {np.sum(y_train_balanced == 0)}")
print(f"High risk (1): {np.sum(y_train_balanced == 1)}")

# Create datasets and loaders like notebook
train_x = torch.Tensor(X_train_balanced).float()
train_y = torch.Tensor(y_train_balanced).long()
train_ds = td.TensorDataset(train_x, train_y)
train_loader = td.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

test_x = torch.Tensor(X_test).float()
test_y = torch.Tensor(y_test).long()
test_ds = td.TensorDataset(test_x, test_y)
test_loader = td.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

print("Ready to load data")

# Define improved neural network like notebook
hl = 64  # Increased number of hidden layer nodes


class ImprovedRiskNet(nn.Module):
    def __init__(self, input_size):
        super(ImprovedRiskNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hl)
        self.bn1 = nn.BatchNorm1d(hl)
        self.dropout1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hl, hl // 2)
        self.bn2 = nn.BatchNorm1d(hl // 2)
        self.dropout2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(hl // 2, hl // 4)
        self.bn3 = nn.BatchNorm1d(hl // 4)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hl // 4, 2)  # 2 classes for binary classification

    def forward(self, x):
        x = self.dropout1(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(torch.relu(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x


# Create model instance
model = ImprovedRiskNet(len(features))
print(model)


# Focal Loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Training functions like notebook
def train(model, data_loader, optimizer, criterion):
    model.train()
    train_loss = 0

    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / (batch + 1)
    print(f"Training set: Average loss: {avg_loss:.6f}")
    return avg_loss


def test(model, data_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        batch_count = 0
        for batch, tensor in enumerate(data_loader):
            batch_count += 1
            data, target = tensor
            out = model(data)
            test_loss += criterion(out, target).item()
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()

            all_predictions.extend(predicted.numpy())
            all_labels.extend(target.numpy())

    avg_loss = test_loss / batch_count
    accuracy = 100.0 * correct / len(data_loader.dataset)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    print(
        f"Validation set: Average loss: {avg_loss:.6f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.0f}%), F1: {f1:.4f}\n"
    )
    return avg_loss, f1


# Specify loss criteria and optimizer like notebook
loss_criteria = FocalLoss(alpha=1, gamma=2)
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5
)

# Track metrics like notebook
epoch_nums = []
training_loss = []
validation_loss = []
validation_f1 = []

# Train over 100 epochs like notebook with early stopping
epochs = 100
best_f1 = 0.0
patience_counter = 0

for epoch in range(1, epochs + 1):
    print(f"Epoch: {epoch}")
    train_loss = train(model, train_loader, optimizer, loss_criteria)
    test_loss, f1 = test(model, test_loader, loss_criteria)

    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    validation_f1.append(f1)

    # Learning rate scheduling
    scheduler.step(test_loss)

    # Early stopping based on F1 score
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "best_improved_notebook_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= 20:  # Early stopping patience
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load("best_improved_notebook_model.pt"))

# Plot training history like notebook
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(["training", "validation"], loc="upper right")
plt.title("Training and Validation Loss")

plt.subplot(1, 3, 2)
plt.plot(epoch_nums, validation_f1, color="green")
plt.xlabel("epoch")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score")

plt.subplot(1, 3, 3)
# Calculate accuracy from F1 scores for visualization
accuracies = [100 * (1 - loss) for loss in validation_loss]
plt.plot(epoch_nums, accuracies, color="orange")
plt.xlabel("epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy")

plt.tight_layout()
plt.savefig(
    "improved_notebook_style_training_history.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Evaluate model
model.eval()
with torch.no_grad():
    x = torch.Tensor(X_test).float()
    outputs = model(x)
    probabilities = torch.softmax(outputs, dim=1)
    _, predicted = torch.max(outputs.data, 1)

# Calculate metrics
accuracy = (predicted.numpy() == y_test).mean()
print(f"Final Accuracy: {accuracy:.4f}")

# Calculate F1 score
f1 = f1_score(y_test, predicted.numpy(), average="weighted")
print(f"Final F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, predicted.numpy())
print("Confusion Matrix:")
print(cm)

# Calculate cost
cost_matrix = np.array([[0, 50], [5, 0]])  # [actual][predicted]
cost = np.sum(cm * cost_matrix)
print(f"Total Cost: {cost:.2f}")

# Cost breakdown
print("Cost Breakdown:")
print(f"  True Negatives (correct low risk): {cm[0,0]} (cost: 0)")
print(f"  False Positives (incorrect high risk): {cm[0,1]} (cost: {cm[0,1] * 50})")
print(f"  False Negatives (incorrect low risk): {cm[1,0]} (cost: {cm[1,0] * 5})")
print(f"  True Positives (correct high risk): {cm[1,1]} (cost: 0)")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test, predicted.numpy(), target_names=["Low Risk", "High Risk"]
    )
)

# ROC curve
y_pred_proba = probabilities[:, 1].numpy()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"Improved Notebook Style Model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Improved Notebook Style Model ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("improved_notebook_style_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Save model like notebook
torch.save(model.state_dict(), "improved_notebook_style_risk_model.pt")
print("Improved model saved as improved_notebook_style_risk_model.pt")

print("\n=== Improved Notebook Style Analysis Complete ===")
print(f"Improved Notebook Style Model achieved:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  AUC Score: {auc:.4f}")
print(f"  Total Cost: {cost:.2f}")
