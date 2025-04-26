import numpy as np
from torchvision.datasets import MNIST
from collections import defaultdict

def load_mnist_data(dim):
    mnist_train = MNIST(root='./data', train=True, download=True)
    mnist_test = MNIST(root='./data', train=False, download=True)

    all_images = np.concatenate([mnist_train.data.numpy(), mnist_test.data.numpy()])
    all_labels = np.concatenate([mnist_train.targets.numpy(), mnist_test.targets.numpy()])

    class_indices = defaultdict(list)
    for idx, label in enumerate(all_labels):
        class_indices[label].append(idx)

    selected_indices = []
    for label in range(10):
        indices = class_indices[label]
        selected_indices.extend(np.random.choice(indices, 100, replace=False))

    # randomly selected
    X_selected = all_images[selected_indices]
    y_selected = all_labels[selected_indices]

    # reshape
    if dim == 1:
        X_selected = X_selected.reshape(X_selected.shape[0], -1)
    elif dim == 2:
        X_selected = X_selected.reshape(X_selected.shape[0], 28, 28)
    else:
        raise ValueError("dim must be either 1 or 2")

    # shuffle
    indices = np.arange(len(X_selected))
    np.random.shuffle(indices)
    X_selected = X_selected[indices]
    y_selected = y_selected[indices]

    # split
    X_train, y_train = X_selected[:800], y_selected[:800]
    X_test, y_test = X_selected[800:], y_selected[800:]

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # dataloader test
    dim = 2
    X_train, y_train, X_test, y_test = load_mnist_data(dim)
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)
