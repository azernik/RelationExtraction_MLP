import os
import random
import numpy as np
import torch


def set_seed(seed_value=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed_value: The seed value to use (default is 42).
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_directory_exists(file_path):
    """
    Ensures the directory for the given file path exists. If it doesn't, creates it.

    Args:
        file_path: Path to the file.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)