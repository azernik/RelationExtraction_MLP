import torch
import torch.nn as nn
import numpy as np


class MultiLabelMLP(nn.Module):
    """
    Multi-label classification model using a simple MLP architecture.

    Args:
        input_size: Dimensionality of the input features.
        output_size: Number of output classes.
        dropout_rate: Dropout rate for regularization.
    """
    def __init__(self, input_size, output_size, dropout_rate=0.1):
        super(MultiLabelMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class ModelWrapper:
    """
    A wrapper for managing training, validation, and evaluation of a PyTorch model.

    Args:
        model: The PyTorch model to train and evaluate.
        criterion: Loss function.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler (optional).
    """
    def __init__(self, model, criterion, optimizer, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, train_loader, val_loader, num_epochs, patience):
        """
        Train the model with early stopping based on validation loss.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            num_epochs: Maximum number of epochs to train.
            patience: Early stopping patience.
        """
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation phase
            val_loss = 0.0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

            # Learning rate scheduler step
            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for the dataset.

        Returns:
            Loss and evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_outputs = np.vstack(all_outputs)
        all_labels = np.vstack(all_labels)

        # Compute metrics
        accuracy = np.mean((all_outputs > 0.5) == all_labels)
        precision, recall, f1, threshold = self._compute_metrics(all_outputs, all_labels)

        return total_loss, accuracy, precision, recall, f1, threshold

    def _compute_metrics(self, outputs, labels):
        """
        Compute precision, recall, F1, and optimal threshold for multi-label outputs.

        Args:
            outputs: Model outputs (probabilities).
            labels: Ground truth labels.

        Returns:
            precision, recall, F1 score, and the optimal threshold.
        """
        best_f1 = 0.0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (outputs >= threshold).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return precision, recall, best_f1, best_threshold
