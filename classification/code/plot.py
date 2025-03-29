import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os
import argparse
SAVE_PATH = 'trained_models/'
LOG_PATH = 'log.txt'
MODEL_NAME = 'cnn'
# Plot the training and validation loss using this python script

def main(type = None):
    if type not in ['loss', 'acc', 'val_loss']:
        raise ValueError("The type must be one of 'loss' or 'acc'.")
# Load the data fr
    path_cnn = os.path.join(SAVE_PATH, 'log', 'log_cnn.pth.csv')
    path_mlp = os.path.join(SAVE_PATH, 'log', 'log_mlp.pth.csv')

    data_cnn = pd.read_csv(path_cnn)
    data_mlp = pd.read_csv(path_mlp)

    # Plot the training loss of cnn model and mlp model in the same figure
    if type == 'loss':
        plt.figure(figsize=(10, 5))
        plt.plot(data_cnn['train_loss'], label='CNN Training Loss')
        plt.plot(data_mlp['train_loss'], label='MLP Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss of CNN and MLP')
        plt.legend()
        plt.show()
    elif type == 'acc':
        plt.figure(figsize=(10, 5))
        plt.plot(data_cnn['val_accuracy'], label='CNN Validation Accuracy')
        plt.plot(data_mlp['val_accuracy'], label='MLP Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy of CNN and MLP')
        plt.legend()
        plt.show()
    elif type == 'val_loss':
        plt.figure(figsize=(10, 5))
        plt.plot(data_cnn['val_loss'], label='CNN Validation Loss')
        plt.plot(data_mlp['val_loss'], label='MLP Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss of CNN and MLP')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training different classifiers.")

    parser.add_argument("--type", type=str, default="loss", help="MUST be one of 'loss', 'val_loss' or 'acc'.")
    args = parser.parse_args()

    main(
        type=args.type
    )
