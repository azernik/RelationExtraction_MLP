import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split


def preprocess_data(df):
    """
    Preprocess the input DataFrame by splitting 'CORE RELATIONS' into lists
    and removing 'none' labels.
    """
    df['CORE RELATIONS'] = df['CORE RELATIONS'].apply(lambda x: x.split())
    df['CORE RELATIONS'] = df['CORE RELATIONS'].apply(lambda x: [label for label in x if label != 'none'])
    return df


def tokens_to_embedding(tokens, wv, embedding_dim):
    """
    Convert a list of tokens into an embedding vector by averaging token embeddings.
    """
    token_embeddings = []
    for token in tokens:
        if token in wv:
            token_embeddings.append(wv[token])
        else:
            token_embeddings.append(np.zeros(embedding_dim))  # If word not in embedding, append a zero vector
    return np.mean(token_embeddings, axis=0)


def build_vocab(token_lists):
    """
    Build a vocabulary from a list of tokenized sentences.
    """
    counter = Counter([word for tokens in token_lists for word in tokens])
    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), start=2)}  # Reserve 0 for padding, 1 for unknown
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab


def split_data(df, test_size=0.2, val_size=0.5, random_state=42):
    """
    Split the data into training, validation, and test sets.
    """
    df_train, df_tmp = train_test_split(df, test_size=test_size, random_state=random_state)
    df_val, df_test = train_test_split(df_tmp, test_size=val_size, random_state=random_state)
    return df_train, df_val, df_test
