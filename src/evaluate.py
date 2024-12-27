import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def evaluate_model(model_wrapper, data_loader):
    """
    Evaluate the model on a dataset.

    Args:
        model_wrapper: The trained ModelWrapper instance.
        data_loader: DataLoader for the dataset to evaluate.

    Returns:
        Tuple of loss, accuracy, precision, recall, F1 score, and optimal threshold.
    """
    print("Evaluating the model...")
    total_loss, accuracy, precision, recall, f1, best_threshold = model_wrapper.evaluate(data_loader)
    return total_loss, accuracy, precision, recall, f1, best_threshold


def predict_and_save(model_wrapper, test_df, vectorizer, wv, mlb, embedding_dim, batch_size, output_path):
    """
    Make predictions on the test set and save them to a CSV file.

    Args:
        model_wrapper: Trained ModelWrapper instance.
        test_df: DataFrame containing test data.
        vectorizer: Trained CountVectorizer instance.
        wv: Pre-trained word vector model.
        mlb: MultiLabelBinarizer instance.
        embedding_dim: Dimensionality of embeddings.
        batch_size: Batch size for prediction.
        output_path: File path to save predictions.
    """
    print("Generating predictions on test data...")
    
    # Tokenize and compute embeddings
    test_df['tokens'] = test_df['UTTERANCES'].apply(lambda x: word_tokenize(x.lower()))
    x_test_glove = np.array(test_df['tokens'].apply(lambda tokens: tokens_to_embedding(tokens, wv, embedding_dim)).tolist())
    x_test_bow = vectorizer.transform(test_df['UTTERANCES']).toarray()

    # Combine features
    x_test_combined = np.concatenate((x_test_bow, x_test_glove), axis=1)
    x_test_tensor = torch.tensor(x_test_combined, dtype=torch.float32)

    # Create DataLoader
    test_dataset = TensorDataset(x_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make predictions
    test_preds = model_wrapper.predict(test_loader, test_df, mlb)

    # Save predictions
    print(f"Saving predictions to {output_path}...")
    test_preds.to_csv(output_path, index=False)
