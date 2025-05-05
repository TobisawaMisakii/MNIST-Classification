import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
from model_train import train_model, plot_metrics, save_results
from data_preprocess import load_mnist_data

os.makedirs('results', exist_ok=True)


def data_load_cnn():
    #加载数据并转换为torch tensor
    X_train, y_train, X_test, y_test = load_mnist_data(dim=2)

    X_train_cnn = torch.FloatTensor(X_train) / 255.0
    X_test_cnn = torch.FloatTensor(X_test) / 255.0
    y_train_cnn = torch.LongTensor(y_train.astype(int))
    y_test_cnn = torch.LongTensor(y_test.astype(int))

    train_dataset = TensorDataset(X_train_cnn, y_train_cnn)
    test_dataset = TensorDataset(X_test_cnn, y_test_cnn)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_and_evaluate():
    train_loader, test_loader = data_load_cnn()
    kernel_sizes = [3, 5, 7]
    cnn_results = {}

    filename = f'cnn_results.txt'
    os.remove(f'results/{filename}') if os.path.exists(f'results/{filename}') else None

    for ks in kernel_sizes:
        print(f"\nTraining with kernel size: {ks}")
        accuracies = []
        best_test_acc = 0
        best_epoch = 0

        for run in range(3):
            model = CNN(kernel_size=ks)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            print(f"\nRun {run + 1}/3")
            train_losses, test_losses, train_accs, test_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, epochs=25)

            # 记录最佳测试准确率及其epoch
            current_best_acc = max(test_accs)
            if current_best_acc > best_test_acc:
                best_test_acc = current_best_acc
                best_epoch = test_accs.index(best_test_acc) + 1

            accuracies.append(max(test_accs))

            # 绘制曲线
            if run == 2:
                plot_metrics(f"CNN (Kernel={ks})", train_losses, test_losses, train_accs, test_accs)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        cnn_results[ks] = (mean_acc, std_acc, best_epoch)

        result_str = (f"Kernel Size: {ks} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | "
                      f"Best Epoch: {best_epoch}")
        print("\n" + result_str)
        save_results(filename, result_str)
        save_results(filename, "-" * 50)

    print(f"\nAll results saved to: results/{filename}")


if __name__ == "__main__":
    train_and_evaluate()