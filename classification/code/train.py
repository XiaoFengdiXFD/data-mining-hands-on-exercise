import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

IMAGE_DIR = 'dataset_images/'
SAVE_PATH = 'trained_models/'
LOG_PATH = 'log.txt'

# Construct the MLP model class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
# Construct the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_dnn(model: nn.Module, train_loader, val_loader, criterion, optimizer, epochs=10,save_path=SAVE_PATH,model_name ='unspecified', log_path='LOG_PATH'):
    train_losses = []
    val_losses = []
    val_accuracies  = []

    print("Training started...")
    with open(log_path, 'w') as log_file:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.view(-1, 784) if isinstance(model, MLP) else inputs.view(-1, 1, 28, 28)  # 根据模型类型 reshape
                # Move the inputs and labels to the device
                device = model.parameters().__next__().device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # Record the training loss for each epoch
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validate the model on the test set
            val_loss, val_accuracy = evaluate_dnn(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            #print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            # Log results
            log_str = f'Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n'
            print(log_str.strip())
            log_file.write(log_str)

    # Save the model
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    print(f"Model saved to {save_path}")

    # Save the training loss and validation accuracy as a csv file.
    # The column of the csv file should be 'epoch', 'train_loss', 'val_loss', 'val_accuracy'
    log_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    })
    # Save the data to <model_path>/log/log_<model_name>.csv
    filename = f"log_{model_name}.csv"
    save_dir = os.path.join(save_path, 'log')
    os.makedirs(save_dir, exist_ok=True)
    log_df.to_csv(os.path.join(save_path, 'log', filename), index=False)

    
    # Plot the curves
    plt.figure(figsize=(12, 5))

    # Plot the training loss
    plt.subplot(2, 1, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot the validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



def evaluate_dnn(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.view(-1, 784) if isinstance(model, MLP) else inputs.view(-1, 1, 28, 28)
            # Move the inputs and labels to the device
            device = model.parameters().__next__().device
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy

def save_images(features, labels=None, image_dir=IMAGE_DIR):
    for idx in range(features.shape[0]):
        img_array = (features[idx].reshape(28, 28) * 255).astype(np.uint8)  # 还原归一化，并转为uint8类型
        img = Image.fromarray(img_array)

        # 生成文件名
        if labels is not None:
            label = labels[idx]
            filename = os.path.join(image_dir, f"img_{idx}_label_{label}.png")
        else:
            filename = os.path.join(image_dir, f"img_{idx}.png")

        img.save(filename)


def main(
        classifer: str = None,
        epochs: int = 20
    ):
    if classifer not in ["mlp", "cnn", "svm"]:
        raise ValueError("The classifier must be one of 'mlp', 'cnn' or 'svm'.")
    
    # Read the data
    if classifer == "mlp" or classifer == "svm":
        # For SVM and MLP classifiers, we use all the 784 pixels as features
        train_data = pd.read_csv("training.csv")
        test_data = pd.read_csv("testing.csv")
    else:
        train_data = pd.read_csv("training.csv")
        # For CNN classifier, we can also use the reshaped 28x28 pixels as features
        # To be implemented

    print(f"Training the classifier: {classifer}")

    # Split the data into features and labels
    X = train_data.iloc[:, 1:].values.astype("float32")/255.0 # Normalize the data
    y = train_data['label'].values

    # Save the images if the classifier is CNN
    if classifer == "cnn":
        save_images(X, y)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a custom dataset
    class DigitDataset(Dataset):
        def __init__(self, features, labels=None):
            self.features = torch.tensor(features)
            self.labels = torch.tensor(labels).long() if labels is not None else None

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            if self.labels is not None:
                return self.features[idx], self.labels[idx]
            else:
                return self.features[idx]

    # Create the dataloaders
    train_dataset = DigitDataset(X_train, y_train)
    val_dataset = DigitDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("Data loaded successfully.")

    # Instantiate the model
    if classifer == "mlp":
        model = MLP()
    elif classifer == "cnn":
        model = CNN()
    else:
        svm_classifier = svm.SVC(kernel='rbf', C=10.0, gamma='scale') # RBF kernel with regularization parameter C

    # Define the loss function and optimizer
    if classifer in ["mlp", "cnn"]:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Model instantiated successfully.")

    # Train the model
    if classifer in ["mlp", "cnn"]:
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        model.to(device)
        train_dnn(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, model_name=classifer+".pth")
    else:
        # Fit the model
        svm_classifier.fit(X_train, y_train)

        # Validate the model
        y_val_pred = svm_classifier.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {val_accuracy:.4f}")

        # Save the model
        model_path = 'svm_model.pkl'
        joblib.dump(svm_classifier, model_path)
        print(f"Model saved to {model_path}")

        # Save the log
        log_path = 'svm_log.txt'
        with open(log_path, 'w') as log_file:
            log_file.write(f"Training the classifier: {classifer}\n")
            log_file.write(f"Model: SVM with RBF kernel, C=10.0, gamma='scale'\n")
            log_file.write(f"SVM Validation Accuracy: {val_accuracy:.4f}\n")
            print(f"Training log saved to {log_path}")
        
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training different classifiers.")

    parser.add_argument("--classifier", type=str, default="mlp", help="MUST be one of 'svm', 'mlp' or 'cnn'.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train the model.")    
    args = parser.parse_args()

    main(
        classifer=args.classifier,
        epochs=args.epochs
    )
