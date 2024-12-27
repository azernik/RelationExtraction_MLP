import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ModelWrapper, MultiLabelMLP


def train_model(train_loader, val_loader, input_size, output_size, dropout_rate, learning_rate, num_epochs, patience):
    """
    Train the MultiLabelMLP model using the provided data loaders.
    
    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        input_size: Dimensionality of the input features.
        output_size: Number of output classes.
        dropout_rate: Dropout rate for the model.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Maximum number of epochs for training.
        patience: Early stopping patience.

    Returns:
        Trained ModelWrapper instance.
    """
    # Initialize model, loss function, optimizer, and scheduler
    model = MultiLabelMLP(input_size, output_size, dropout_rate=dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = None  # Can be updated if learning rate scheduling is needed

    # Wrap the model with additional training/evaluation methods
    model_wrapper = ModelWrapper(model, criterion, optimizer, scheduler)

    # Train the model
    print("Starting training...")
    model_wrapper.train(train_loader, val_loader, num_epochs=num_epochs, patience=patience)

    return model_wrapper
