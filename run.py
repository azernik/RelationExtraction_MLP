import argparse
import yaml
import nltk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api
from src.data_processing import preprocess_data, tokens_to_embedding, build_vocab, split_data
from src.train import train_model
from src.evaluate import evaluate_model, predict_and_save
from src.utils import set_seed, ensure_directory_exists

# Download necessary NLTK data
nltk.download('punkt')

def load_config(config_path):
    """
    Load the configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of hyperparameters and settings.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(train_data_path, test_data_path, output_path, config):
    # Set random seed for reproducibility
    set_seed(config['hyperparameters']['random_seed'])

    # DATA PROCESSING SECTION
    print("Loading and preprocessing data...")
    df = pd.read_csv(train_data_path)
    df = preprocess_data(df)

    # Tokenize and build vocab
    df['tokens'] = df['UTTERANCES'].apply(lambda x: nltk.word_tokenize(x.lower()))
    vocab = build_vocab(df['tokens'])

    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    wv = api.load('glove-wiki-gigaword-100')
    embedding_dim = config['hyperparameters']['embedding_dim']
    df['embedding'] = df['tokens'].apply(lambda tokens: tokens_to_embedding(tokens, wv, embedding_dim))

    # Split data into train, val, and test sets
    df_train, df_val, df_test = split_data(df)

    # Bag-of-words features
    vectorizer = CountVectorizer()
    x_train_bow = vectorizer.fit_transform(df_train['UTTERANCES']).toarray()
    x_val_bow = vectorizer.transform(df_val['UTTERANCES']).toarray()
    x_test_bow = vectorizer.transform(df_test['UTTERANCES']).toarray()

    # GloVe embeddings
    x_train_glove = np.array(df_train['embedding'].tolist())
    x_val_glove = np.array(df_val['embedding'].tolist())
    x_test_glove = np.array(df_test['embedding'].tolist())

    # Combine features
    x_train_combined = np.concatenate((x_train_bow, x_train_glove), axis=1)
    x_val_combined = np.concatenate((x_val_bow, x_val_glove), axis=1)
    x_test_combined = np.concatenate((x_test_bow, x_test_glove), axis=1)

    # Encode labels with MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train['CORE RELATIONS'])
    y_val = mlb.transform(df_val['CORE RELATIONS'])
    y_test = mlb.transform(df_test['CORE RELATIONS'])

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train_combined, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val_combined, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_combined, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create PyTorch datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # TRAINING SECTION
    print("Training the model...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: np.random.seed(config['hyperparameters']['random_seed'])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=False
    )

    # Model parameters
    input_size = x_train_combined.shape[1]
    output_size = len(mlb.classes_)

    # Train the model
    model_wrapper = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        output_size=output_size,
        dropout_rate=config['hyperparameters']['dropout_rate'],
        learning_rate=config['hyperparameters']['learning_rate'],
        num_epochs=config['hyperparameters']['num_epochs'],
        patience=config['hyperparameters']['patience']
    )

    # EVALUATION SECTION
    print("Evaluating the model...")
    test_loss, accuracy, precision, recall, f1, best_threshold = evaluate_model(
        model_wrapper=model_wrapper,
        data_loader=test_loader
    )

    print(f"\nBest Threshold: {best_threshold}")
    print(f"\nFinal Evaluation on Test Set:\nTest Loss: {test_loss:.4f}\n"
          f"Test Accuracy: {accuracy:.4f}\nTest Precision: {precision:.4f}\n"
          f"Test Recall: {recall:.4f}\nTest F1: {f1:.4f}")

    # TESTING SECTION
    ensure_directory_exists(output_path)

    print("Generating predictions for test data...")
    test_df = pd.read_csv(test_data_path)
    predict_and_save(
        model_wrapper=model_wrapper,
        test_df=test_df,
        vectorizer=vectorizer,
        wv=wv,
        mlb=mlb,
        embedding_dim=embedding_dim,
        batch_size=config['hyperparameters']['batch_size'],
        output_path=output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training, testing, and inference for relation extraction.")
    parser.add_argument('train_data', type=str, help="Path to training data CSV.")
    parser.add_argument('test_data', type=str, help="Path to test data CSV.")
    parser.add_argument('output', type=str, help="Path to save test predictions CSV.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    main(args.train_data, args.test_data, args.output, config)
