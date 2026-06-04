import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True):
    n = len(X)
    idx = np.arange(n)

    if shuffle:
        np.random.shuffle(idx)

    split = int(n * (1 - test_size))

    train_idx = idx[:split]
    test_idx = idx[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]