from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import load_mnist_data
import numpy as np
import os
from model_train import save_results

os.makedirs('results', exist_ok=True)

def data_load_svm():
    X_train, y_train, X_test, y_test = load_mnist_data(dim=1)

    # 标准化
    scaler = MinMaxScaler()
    X_train_svm = scaler.fit_transform(X_train)
    X_test_svm = scaler.transform(X_test)

    return X_train_svm, y_train, X_test_svm, y_test


def evaluate_svm_model(X_train, y_train, X_test, y_test, kernel, random_state=42):
    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred)


def train_and_evaluate():
    X_train, y_train, X_test, y_test = data_load_svm()
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    svm_results = {}
    filename = f'svm_results.txt'

    for kernel in kernels:
        print(f"\nEvaluating kernel: {kernel}")
        accuracies = []

        for run in range(5):
            acc = evaluate_svm_model(X_train, y_train, X_test, y_test,
                                     kernel, random_state=42 + run)
            accuracies.append(acc)
            print(f"Run {run + 1}/5 - Accuracy: {acc:.4f}")

        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        svm_results[kernel] = (mean_acc, std_acc)

        # 保存结果
        result_str = f"Kernel: {kernel:8} | Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}"
        print("\n" + result_str)
        save_results(filename, result_str)
        save_results(filename, "-" * 50)

    save_results(filename, "\nSummary:")
    best_kernel = max(svm_results.items(), key=lambda x: x[1][0])
    save_results(filename,
                 f"Best Kernel: {best_kernel[0]} (Accuracy: {best_kernel[1][0]:.4f} ± {best_kernel[1][1]:.4f})")

    print(f"\nAll results saved to: results/{filename}")
    return svm_results


if __name__ == "__main__":
    results = train_and_evaluate()