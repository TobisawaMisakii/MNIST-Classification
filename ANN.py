import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from data_preprocess import load_mnist_data
from model_train import train_model, plot_metrics, save_results

os.makedirs('results', exist_ok=True)

def data_load_ann():
    X_train, y_train, X_test, y_test = load_mnist_data(dim=1)

    X_train_ann = torch.FloatTensor(X_train) / 255.0
    X_test_ann = torch.FloatTensor(X_test) / 255.0
    y_train_ann = torch.LongTensor(y_train.astype(int))
    y_test_ann = torch.LongTensor(y_test.astype(int))

    train_dataset = TensorDataset(X_train_ann, y_train_ann)
    test_dataset = TensorDataset(X_test_ann, y_test_ann)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_and_evaluate():
    train_loader, test_loader = data_load_ann()
    learning_rates = [0.0005, 0.001, 0.01, 0.1]
    ann_results = {}

    filename = f'ann_results.txt'
    if os.path.exists(f'results/{filename}'):
        os.remove(f'results/{filename}')

    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        accuracies = []
        best_test_acc = 0
        best_epoch = 0

        for run in range(3):
            model = ANN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            print(f"\nRun {run + 1}/3")
            train_losses, test_losses, train_accs, test_accs = train_model(
                model, train_loader, test_loader, criterion, optimizer, epochs=40)

            # 记录最佳测试准确率及其epoch
            current_best_acc = max(test_accs)
            if current_best_acc > best_test_acc:
                best_test_acc = current_best_acc
                best_epoch = test_accs.index(best_test_acc) + 1

            accuracies.append(max(test_accs))

            if run == 2:
                plot_metrics(f"ANN (LR={lr})", train_losses, test_losses, train_accs, test_accs)

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        ann_results[lr] = (mean_acc, std_acc, best_epoch)

        result_str = (f"Learning Rate: {lr} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f} | "
                      f"Best Epoch: {best_epoch}")
        print("\n" + result_str)
        save_results(filename, result_str)
        save_results(filename, "-" * 50)

    print(f"\nAll results saved to: results/{filename}")


if __name__ == "__main__":
    train_and_evaluate()