#!/usr/bin/env python3
"""
Simple PyTorch Neural Network for Payment Risk Classification
==========================================================

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleRiskDataset(Dataset):
    """Simple dataset for payment risk data."""

    def __init__(self, features, labels):
        # Convert to numpy arrays and ensure float32
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleRiskNeuralNetwork(nn.Module):
    """Simple neural network for payment risk classification."""

    def __init__(self, input_size, hidden_size=64):
        super(SimpleRiskNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class SimplePyTorchRiskClassifier:
    """Simple PyTorch-based risk classifier."""

    def __init__(self, sample_size=5000):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sample_size = sample_size
        self.device = device

        # Cost matrix for evaluation
        self.cost_matrix = np.array([[0, 50], [5, 0]])  # [actual][predicted]

    def load_data(self, file_path):
        """Load and sample the dataset."""
        print("Loading dataset...")
        self.data = pd.read_csv(file_path, sep="\t")
        print(f"Original dataset shape: {self.data.shape}")

        # Sample the data for faster processing
        if len(self.data) > self.sample_size:
            # Stratified sampling to maintain class distribution
            high_risk = self.data[self.data["CLASS"] == "yes"]
            low_risk = self.data[self.data["CLASS"] == "no"]

            # Calculate sample sizes
            high_risk_sample = min(len(high_risk), int(self.sample_size * 0.06))
            low_risk_sample = self.sample_size - high_risk_sample

            # Sample each class
            high_risk_sampled = high_risk.sample(n=high_risk_sample, random_state=42)
            low_risk_sampled = low_risk.sample(n=low_risk_sample, random_state=42)

            # Combine samples
            self.data = pd.concat([high_risk_sampled, low_risk_sampled]).sample(
                frac=1, random_state=42
            )
            print(f"Sampled dataset shape: {self.data.shape}")

        print(f"Class distribution:")
        print(self.data["CLASS"].value_counts())

        return self.data

    def preprocess_data(self):
        """Simple data preprocessing for PyTorch."""
        print("\n=== Simple PyTorch Data Preprocessing ===")

        # Create a copy for preprocessing
        df = self.data.copy()

        # Remove ORDER_ID
        df = df.drop("ORDER_ID", axis=1)

        # Handle missing values
        print("Handling missing values...")

        # Fill missing values
        df = df.fillna("unknown")

        # Convert numerical columns
        numerical_cols = [
            "VALUE_ORDER",
            "AMOUNT_ORDER",
            "SESSION_TIME",
            "AMOUNT_ORDER_PRE",
            "VALUE_ORDER_PRE",
        ]
        for col in numerical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Create simple risk features
        print("Creating simple risk features...")

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
            if col in df.columns:
                df[col] = (df[col] == "yes").astype(int)

        # Payment method encoding
        if "Z_METHODE" in df.columns:
            df["Z_METHODE"] = (
                df["Z_METHODE"]
                .map({"credit_card": 1, "check": 2, "debit_note": 3, "?": 0})
                .fillna(0)
            )

        # Card type encoding
        if "Z_CARD_ART" in df.columns:
            df["Z_CARD_ART"] = (
                df["Z_CARD_ART"]
                .map({"Visa": 1, "Eurocard": 2, "Mastercard": 3, "?": 0})
                .fillna(0)
            )

        # Weekday encoding
        if "WEEKDAY_ORDER" in df.columns:
            weekday_map = {
                "Monday": 1,
                "Tuesday": 2,
                "Wednesday": 3,
                "Thursday": 4,
                "Friday": 5,
                "Saturday": 6,
                "Sunday": 7,
            }
            df["WEEKDAY_ORDER"] = df["WEEKDAY_ORDER"].map(weekday_map).fillna(0)

        # Customer type
        if "NEUKUNDE" in df.columns:
            df["NEUKUNDE"] = (df["NEUKUNDE"] == "yes").astype(int)

        # Create composite features
        df["RISK_FACTORS"] = df[
            ["CHK_LADR", "CHK_RADR", "CHK_KTO", "CHK_CARD", "CHK_COOKIE", "CHK_IP"]
        ].sum(axis=1)
        df["ADDRESS_FAILURES"] = df[
            [
                "FAIL_LPLZ",
                "FAIL_LORT",
                "FAIL_LPLZORTMATCH",
                "FAIL_RPLZ",
                "FAIL_RORT",
                "FAIL_RPLZORTMATCH",
            ]
        ].sum(axis=1)

        # Drop original columns that were transformed
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
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Drop ANUMMER columns (item numbers)
        anummer_cols = [col for col in df.columns if col.startswith("ANUMMER")]
        df = df.drop(anummer_cols, axis=1)

        # Separate features and target
        X = df.drop("CLASS", axis=1)
        y = df["CLASS"]

        # Ensure all features are numerical
        for col in X.columns:
            if X[col].dtype == "object":
                X[col] = pd.Categorical(X[col]).codes

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Encode target
        y_encoded = self.label_encoder.fit_transform(y)

        print(f"Final feature count: {X_scaled.shape[1]}")

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")

        return X_scaled, y_encoded

    def create_data_loaders(self, batch_size=32):
        """Create PyTorch data loaders."""
        # Create datasets
        train_dataset = SimpleRiskDataset(self.X_train, self.y_train)
        test_dataset = SimpleRiskDataset(self.X_test, self.y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_model(self, train_loader, test_loader, epochs=20, learning_rate=0.001):
        """Train the PyTorch neural network."""
        print(f"\n=== Training Simple PyTorch Neural Network ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")

        # Initialize model
        input_size = self.X_train.shape[1]
        self.model = SimpleRiskNeuralNetwork(input_size).to(self.device)

        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        train_losses = []
        test_accuracies = []

        print("\nTraining progress:")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Evaluation phase
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = self.model(batch_features).squeeze()
                    predicted = (outputs > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            accuracy = correct / total

            train_losses.append(avg_train_loss)
            test_accuracies.append(accuracy)

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}"
                )

        # Plot training history
        self.plot_training_history(train_losses, test_accuracies)

        print(f"\nTraining completed!")
        print(f"Final accuracy: {accuracy:.4f}")

        return self.model

    def plot_training_history(self, train_losses, test_accuracies):
        """Plot training history."""
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(test_accuracies, label="Test Accuracy", color="orange")
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("simple_pytorch_training_history.png", dpi=300, bbox_inches="tight")
        plt.show()

    def evaluate_model(self):
        """Evaluate the trained PyTorch model."""
        print(f"\n=== Simple PyTorch Model Evaluation ===")

        self.model.eval()

        # Convert test data to tensor
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(self.y_test).to(self.device)

        # Make predictions
        with torch.no_grad():
            y_pred_proba = self.model(X_test_tensor).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

        # Convert back to original labels
        y_test_original = self.label_encoder.inverse_transform(self.y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)

        # Calculate metrics
        accuracy = (y_pred == self.y_test).mean()
        auc = roc_auc_score(self.y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test_original, y_pred_original)
        cost = np.sum(cm * self.cost_matrix)

        print("Confusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(classification_report(y_test_original, y_pred_original))

        print(f"\nTotal Cost: {cost:.2f}")

        # Cost breakdown
        print("Cost Breakdown:")
        print(f"  True Negatives (correct low risk): {cm[0,0]} (cost: 0)")
        print(
            f"  False Positives (incorrect high risk): {cm[0,1]} (cost: {cm[0,1] * 50})"
        )
        print(
            f"  False Negatives (incorrect low risk): {cm[1,0]} (cost: {cm[1,0] * 5})"
        )
        print(f"  True Positives (correct high risk): {cm[1,1]} (cost: 0)")

        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f"Simple PyTorch Neural Network (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Simple PyTorch Neural Network ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("simple_pytorch_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

        return {"accuracy": accuracy, "auc": auc, "cost": cost, "confusion_matrix": cm}

    def save_model(self, filename="simple_pytorch_risk_model.pth"):
        """Save the PyTorch model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "input_size": self.X_train.shape[1],
            },
            filename,
        )
        print(f"Simple PyTorch model saved as {filename}")

    def run_complete_analysis(self, data_file):
        """Run the complete simple PyTorch analysis pipeline."""
        print("=== Simple PyTorch Payment Risk Classification System ===")

        # Load data
        self.load_data(data_file)

        # Preprocess data
        X, y = self.preprocess_data()

        # Create data loaders
        train_loader, test_loader = self.create_data_loaders(batch_size=64)

        # Train model
        self.train_model(train_loader, test_loader, epochs=20)

        # Evaluate model
        results = self.evaluate_model()

        # Save model
        self.save_model()

        return results


if __name__ == "__main__":
    """Main function to run the simple PyTorch analysis."""
    classifier = SimplePyTorchRiskClassifier(sample_size=5000)
    results = classifier.run_complete_analysis("risk-dataset.txt")

    print("\n=== Simple PyTorch Analysis Complete ===")
    print(f"Simple PyTorch Neural Network achieved:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Total Cost: {results['cost']:.2f}")
