import torch
import matplotlib.pyplot as plt

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=15):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            if model.__class__.__name__ == 'CNN':
                inputs = inputs.view(-1, 1, 28, 28)
            else :
                inputs = inputs.view(-1, 784)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                if model.__class__.__name__ == 'CNN':
                    inputs = inputs.view(-1, 1, 28, 28)
                else :
                    inputs = inputs.view(-1, 784)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

    return train_losses, test_losses, train_accs, test_accs


def plot_metrics(model_name, train_losses, test_losses, train_accs, test_accs):
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig(f'results/{model_name}_metrics.png')


def save_results(filename, content):
    with open(f'results/{filename}', 'a') as f:
        f.write(content + '\n')