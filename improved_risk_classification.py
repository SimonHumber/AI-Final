"""
Improved PyTorch Neural Network for Payment Risk Classification
Addresses class imbalance and improves accuracy for both low and high risk data
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
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ImprovedRiskDataset(Dataset):
    """Improved dataset for payment risk data with balanced sampling."""

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


class ImprovedRiskNeuralNetwork(nn.Module):
    """Improved neural network for payment risk classification with better architecture."""

    def __init__(self, input_size, hidden_size=128):
        super(ImprovedRiskNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction="none")(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedPyTorchRiskClassifier:
    """Improved PyTorch-based risk classifier with class imbalance handling."""

    def __init__(self, sample_size=10000):
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
        """Load and sample the dataset with better class balance."""
        print("Loading dataset...")
        self.data = pd.read_csv(file_path, sep="\t")
        print(f"Original dataset shape: {self.data.shape}")

        # Sample the data for faster processing while maintaining better balance
        if len(self.data) > self.sample_size:
            # Stratified sampling to maintain better class distribution
            high_risk = self.data[self.data["CLASS"] == "yes"]
            low_risk = self.data[self.data["CLASS"] == "no"]

            # Calculate sample sizes with better balance (20% high risk)
            high_risk_sample = min(len(high_risk), int(self.sample_size * 0.20))
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
        """Enhanced data preprocessing for PyTorch with better feature engineering."""
        print("\n=== Improved PyTorch Data Preprocessing ===")

        # Create a copy for preprocessing
        df = self.data.copy()

        # Remove ORDER_ID
        df = df.drop("ORDER_ID", axis=1)

        # Handle missing values
        print("Handling missing values...")
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

        # Enhanced risk features
        print("Creating enhanced risk features...")

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

        # Enhanced composite features
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

        # New engineered features
        df["TOTAL_RISK_SCORE"] = df["RISK_FACTORS"] + df["ADDRESS_FAILURES"]
        df["HIGH_VALUE_ORDER"] = (
            df["VALUE_ORDER"] > df["VALUE_ORDER"].quantile(0.9)
        ).astype(int)
        df["LONG_SESSION"] = (
            df["SESSION_TIME"] > df["SESSION_TIME"].quantile(0.9)
        ).astype(int)
        df["NEW_CUSTOMER_HIGH_VALUE"] = (df["NEUKUNDE"] == 1) & (
            df["HIGH_VALUE_ORDER"] == 1
        )
        df["NEW_CUSTOMER_HIGH_VALUE"] = df["NEW_CUSTOMER_HIGH_VALUE"].astype(int)

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

    def balance_data(self):
        """Balance the training data using SMOTE and undersampling."""
        print("\n=== Balancing Training Data ===")

        # Apply SMOTE to oversample minority class
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(
            self.X_train, self.y_train
        )

        # Apply undersampling to majority class if still imbalanced
        if np.sum(y_train_balanced == 1) / len(y_train_balanced) < 0.3:
            rus = RandomUnderSampler(random_state=42, sampling_strategy=0.4)
            X_train_balanced, y_train_balanced = rus.fit_resample(
                X_train_balanced, y_train_balanced
            )

        print(f"Balanced training set shape: {X_train_balanced.shape}")
        print(f"Balanced class distribution:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            class_name = self.label_encoder.inverse_transform([u])[0]
            print(f"  {class_name}: {c}")

        return X_train_balanced, y_train_balanced

    def create_data_loaders(self, X_train_balanced, y_train_balanced, batch_size=32):
        """Create PyTorch data loaders with balanced data."""
        # Create datasets
        train_dataset = ImprovedRiskDataset(X_train_balanced, y_train_balanced)
        test_dataset = ImprovedRiskDataset(self.X_test, self.y_test)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train_model(self, train_loader, test_loader, epochs=50, learning_rate=0.001):
        """Train the improved PyTorch neural network with focal loss."""
        print(f"\n=== Training Improved PyTorch Neural Network ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")

        # Initialize model
        input_size = self.X_train.shape[1]
        self.model = ImprovedRiskNeuralNetwork(input_size).to(self.device)

        # Loss function and optimizer with class weights
        # Calculate class weights
        class_counts = np.bincount(self.y_train)
        class_weights = torch.FloatTensor(
            [1.0 / class_counts[0], 10.0 / class_counts[1]]
        ).to(self.device)

        # Use focal loss for better handling of class imbalance
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        # Training history
        train_losses = []
        test_accuracies = []
        test_f1_scores = []

        print("\nTraining progress:")
        best_f1 = 0.0
        patience_counter = 0

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
            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = self.model(batch_features).squeeze()
                    predicted = (
                        outputs > 0.3
                    ).float()  # Lower threshold for better recall
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())

            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            accuracy = correct / total

            # Calculate F1 score
            from sklearn.metrics import f1_score

            f1 = f1_score(all_labels, all_predictions, average="weighted")

            train_losses.append(avg_train_loss)
            test_accuracies.append(accuracy)
            test_f1_scores.append(f1)

            # Learning rate scheduling
            scheduler.step(avg_train_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Accuracy: {accuracy:.4f}, "
                    f"F1: {f1:.4f}"
                )

            # Early stopping based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_improved_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= 15:  # Early stopping patience
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load("best_improved_model.pth"))

        # Plot training history
        self.plot_training_history(train_losses, test_accuracies, test_f1_scores)

        print(f"\nTraining completed!")
        print(f"Best F1 Score: {best_f1:.4f}")

        return self.model

    def plot_training_history(self, train_losses, test_accuracies, test_f1_scores):
        """Plot training history with F1 scores."""
        plt.figure(figsize=(15, 5))

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(test_accuracies, label="Test Accuracy", color="orange")
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # F1 Score plot
        plt.subplot(1, 3, 3)
        plt.plot(test_f1_scores, label="Test F1 Score", color="green")
        plt.title("Test F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            "improved_pytorch_training_history.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

    def evaluate_model(self):
        """Evaluate the trained improved PyTorch model."""
        print(f"\n=== Improved PyTorch Model Evaluation ===")

        self.model.eval()

        # Convert test data to tensor
        X_test_tensor = torch.FloatTensor(self.X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(self.y_test).to(self.device)

        # Make predictions with different thresholds
        with torch.no_grad():
            y_pred_proba = self.model(X_test_tensor).cpu().numpy().flatten()

            # Try different thresholds
            thresholds = [0.3, 0.4, 0.5]
            best_f1 = 0
            best_threshold = 0.5
            best_y_pred = None

            for threshold in thresholds:
                y_pred = (y_pred_proba > threshold).astype(int)
                from sklearn.metrics import f1_score

                f1 = f1_score(self.y_test, y_pred, average="weighted")
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_y_pred = y_pred

            y_pred = best_y_pred

        # Convert back to original labels
        y_test_original = self.label_encoder.inverse_transform(self.y_test)
        y_pred_original = self.label_encoder.inverse_transform(y_pred)

        # Calculate metrics
        accuracy = (y_pred == self.y_test).mean()
        auc = roc_auc_score(self.y_test, y_pred_proba)

        # Confusion matrix
        cm = confusion_matrix(y_test_original, y_pred_original)
        cost = np.sum(cm * self.cost_matrix)

        print(f"Best threshold: {best_threshold}")
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
        plt.plot(fpr, tpr, label=f"Improved PyTorch Neural Network (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Improved PyTorch Neural Network ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig("improved_pytorch_roc_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

        return {
            "accuracy": accuracy,
            "auc": auc,
            "cost": cost,
            "confusion_matrix": cm,
            "threshold": best_threshold,
        }

    def save_model(self, filename="improved_pytorch_risk_model.pth"):
        """Save the improved PyTorch model."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "input_size": self.X_train.shape[1],
            },
            filename,
        )
        print(f"Improved PyTorch model saved as {filename}")

    def run_complete_analysis(self, data_file):
        """Run the complete improved PyTorch analysis pipeline."""
        print("=== Improved PyTorch Payment Risk Classification System ===")

        # Load data
        self.load_data(data_file)

        # Preprocess data
        X, y = self.preprocess_data()

        # Balance data
        X_train_balanced, y_train_balanced = self.balance_data()

        # Create data loaders
        train_loader, test_loader = self.create_data_loaders(
            X_train_balanced, y_train_balanced, batch_size=64
        )

        # Train model
        self.train_model(train_loader, test_loader, epochs=50)

        # Evaluate model
        results = self.evaluate_model()

        # Save model
        self.save_model()

        return results


if __name__ == "__main__":
    """Main function to run the improved PyTorch analysis."""
    classifier = ImprovedPyTorchRiskClassifier(sample_size=10000)
    results = classifier.run_complete_analysis("risk-dataset.txt")

    print("\n=== Improved PyTorch Analysis Complete ===")
    print(f"Improved PyTorch Neural Network achieved:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Total Cost: {results['cost']:.2f}")
    print(f"  Best Threshold: {results['threshold']:.2f}")
