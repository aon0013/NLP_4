
import numpy as np
import torch
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def minibatches(data, batch_size):
    """Create minibatches from data"""
    data_size = len(data)
    indices = list(range(data_size))
    random.shuffle(indices)

    for i in range(0, data_size, batch_size):
        batch_indices = indices[i:i + batch_size]
        yield [data[j] for j in batch_indices]


def extract_features(partial_parse):
    """Extract features from a partial parse

    This is a simplified version. The real implementation would
    extract specific features from the stack, buffer, and dependencies.
    """
    # Get the top 3 items from the stack (or PAD if not available)
    stack = partial_parse.stack[-3:] if len(partial_parse.stack) >= 3 else \
        ['PAD'] * (3 - len(partial_parse.stack)) + partial_parse.stack

    # Get the next 3 items from the buffer (or PAD if not available)
    buffer = partial_parse.buffer[:3] if len(partial_parse.buffer) >= 3 else \
        partial_parse.buffer + ['PAD'] * (3 - len(partial_parse.buffer))

    # Convert these words to indices using a vocabulary (simplified)
    # In a real implementation, you'd use a proper vocabulary object
    word_to_id = {'PAD': 0, 'ROOT': 1}  # Simplified vocabulary

    # Create a feature vector from stack and buffer words
    features = []
    for word in stack + buffer:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)
        features.append(word_to_id[word])

    return torch.tensor(features)


def load_and_preprocess_data(debug=False):
    """Load and preprocess data

    This is a simplified version. The real implementation would
    load data from files, process it, and create embeddings.
    """
    # Simplified data loading for demonstration
    # In a real implementation, you'd load from files

    # Generate some random data for demonstration
    vocab_size = 1000
    embed_size = 50

    # Create random embeddings
    embeddings = np.random.randn(vocab_size, embed_size)

    # Create random training, dev, and test data
    data_size = 100 if debug else 10000
    n_features = 36  # Typically we would extract multiple features

    # Generate random word indices, POS indices, and labels
    def generate_random_data(size):
        data = []
        for _ in range(size):
            word_indices = torch.randint(0, vocab_size, (n_features,))
            pos_indices = torch.randint(0, 50, (n_features,))  # Assuming 50 POS tags
            label = random.randint(0, 2)  # 0, 1, 2 for SHIFT, LEFT-ARC, RIGHT-ARC

            # Also generate some random dependencies for evaluation
            deps = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(5)]

            data.append((word_indices, pos_indices, label, deps))
        return data

    train_data = generate_random_data(data_size)
    dev_data = generate_random_data(data_size // 10)
    test_data = generate_random_data(data_size // 10)

    # Create a simple vocabulary (mapping words to indices)
    vocab = {str(i): i for i in range(vocab_size)}

    return train_data, dev_data, test_data, embeddings, vocab