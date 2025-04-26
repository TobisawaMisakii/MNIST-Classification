from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from data_preprocess import load_mnist_data
import numpy as np

X_train, y_train, X_test, y_test = load_mnist_data(dim=1)

# 数据标准化
scaler = MinMaxScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# 不同核函数测试
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = {}

for kernel in kernels:
    accuracies = []
    for _ in range(5):  # 多次运行取平均
        svm = SVC(kernel=kernel, random_state=42)
        svm.fit(X_train_svm, y_train)
        y_pred = svm.predict(X_test_svm)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    svm_results[kernel] = (mean_acc, std_acc)
    print(f"Kernel: {kernel}, Mean Accuracy: {mean_acc:.4f}, Std: {std_acc:.4f}")