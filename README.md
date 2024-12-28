# Relation Extraction from Natural Language Using Deep Learning

## Introduction
This project focuses on predicting knowledge graph relations from user utterances in conversational systems. The task is a multi-label, multi-class classification problem where each utterance may correspond to one or more core relations. The objective was to design and implement a deep learning model capable of accurately identifying these relations while generalizing effectively to unseen data.

**Note:** This project was initially developed as part of a master's course on Deep Learning for Natural Language Processing. A detailed report explaining the development and results can be accessed [here](https://drive.google.com/file/d/1kxhQD5RbO5UFAH-8kc79NtHWsEKg8Gkg/view).

## Final Model Overview

### Model Architecture
The final model is a Multi-Label Multi-Layer Perceptron (MLP) combining two complementary input representations:
1. **Bag-of-Words (BoW):** Captures word frequency information.
2. **GloVe Word Embeddings:** Pre-trained 100-dimensional embeddings that encode semantic relationships between words.

The architecture includes:
- **Input Layer:** A combined feature vector incorporating BoW and GloVe embeddings.
- **Hidden Layers:** Three fully connected layers, with ReLU activations for non-linearity.
- **Regularization:** Dropout (0.1) and Batch Normalization between layers to reduce overfitting and improve training stability.
- **Output Layer:** Sigmoid activation for multi-label classification.

### Hyperparameters
- **Dropout Rate:** 0.1
- **Learning Rate:** 0.0001
- **Batch Size:** 64
- **Number of Epochs:** 200 (with early stopping after 5 epochs of no improvement)

### Optimization
The model was tuned using grid search to find the optimal combination of hyperparameters. Early stopping was employed to prevent overfitting.

## Dataset
- **Size:** 2,312 user utterances related to movies.
- **Labels:** 18 unique knowledge graph relations (e.g., `movie.directed_by`, `movie.starring.actor`).
- **Imbalance:** The most frequent label appears in 15% of the utterances, while others are associated with fewer than five examples.
- **Tokenization:** Inputs were lowercased and tokenized using NLTK.

## Implementation Details
### Repository Structure
```plaintext
src/
├── data_processing.py   # Preprocessing functions
├── model.py             # Model architecture and wrapper
├── train.py             # Training logic
├── evaluate.py          # Evaluation and prediction functions
├── utils.py             # Utility functions (e.g., set_seed)
run.py                   # Main entry point for training and evaluation
config.yaml              # Configuration file for hyperparameters
```

### Key Steps
1. **Preprocessing:**
   - Tokenization of text inputs.
   - Generation of BoW features and GloVe embeddings.
   - Label binarization for multi-label classification.
2. **Training:**
   - Utilized the AdamW optimizer and binary cross-entropy loss.
   - Implemented dropout and early stopping for regularization.
3. **Evaluation:**
   - Metrics include F1 score, accuracy, precision, and recall.
   - Optimal threshold selection for classification.

## Usage
### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```
3. Download required resources:
   - Pre-trained GloVe embeddings (`glove-wiki-gigaword-100`).
   - NLTK tokenization data:
     ```python
     import nltk
     nltk.download('punkt')
     ```

### Running the Project
To train and evaluate the model, use the following command:
```bash
python run.py train_data.csv test_data.csv output.csv --config config.yaml
```
- `train_data.csv`: Path to the training dataset.
- `test_data.csv`: Path to the test dataset.
- `output.csv`: Path to save the predictions.
- `config.yaml`: Configuration file for hyperparameters (default: `config.yaml`).

## Results
The final model achieved the following performance on the test set:
- **F1 Score:** 0.8247
- **Precision:** High precision across the most frequent labels.
- **Recall:** Balanced recall despite the label imbalance.

The combination of BoW and GloVe embeddings proved critical for achieving this level of performance.

## Acknowledgments
- **Libraries:** PyTorch, NLTK, Gensim, Scikit-learn.
- **Dataset:** Provided as part of the course assignment.
- **References:**
  - Pennington, Socher, and Manning. "GloVe: Global Vectors for Word Representation" (2014).
  - Srivastava et al. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014).
  - Manning et al. "Introduction to Information Retrieval" (2008).
