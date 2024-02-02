import os
import pandas as pd

# read the dataset
def read_data(root):

    train_folder_path = os.path.join(root, "train")
    test_folder_path = os.path.join(root, "test")

    train_csv_path = os.path.join(train_folder_path, "_classes.csv")
    test_csv_path = os.path.join(test_folder_path, "_classes.csv")

    train_labels = pd.read_csv(train_csv_path, header=0)
    test_labels = pd.read_csv(test_csv_path, header=0)

    return train_folder_path, test_folder_path, train_labels, test_labels