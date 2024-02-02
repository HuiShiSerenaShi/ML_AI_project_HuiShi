import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
import torch.nn.functional as F
from matplotlib import pyplot as plt


#Create a basic NN classifier with one fully connected layer using PyTorch
class MultiLabelClassifier_FC1(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiLabelClassifier_FC1, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

#Create a basic NN classifier with two fully connected layers using PyTorch
class MultiLabelClassifier_FC2(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiLabelClassifier_FC2, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, num_classes)  
        self.dropout = nn.Dropout(p=0.1)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(self.fc1(x))  # rectified linear unit (ReLU) activation function
        out = self.dropout(out)  # avoid overfitting
        out = self.fc2(out)
        out = self.sigmoid(out) # sigmoid activation function
        return out

# train the model and make predictions
def train_and_pred(features_train, labels_train, features_test, labels_test, MultiLabelClassifier):
    # Initializing compute device (use GPU if available).
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    labels_train_cpu = labels_train
    labels_test_cpu = labels_test

    features_train = torch.tensor(features_train, dtype=torch.float32).to(device)
    labels_train = torch.tensor(labels_train, dtype=torch.float32).to(device)
    features_test = torch.tensor(features_test, dtype=torch.float32).to(device)
    labels_test = torch.tensor(labels_test, dtype=torch.float32).to(device)

    input_size = features_train.shape[1]
    num_classes = 3 
    # Instantiate the model and move it to the selected device
    model = MultiLabelClassifier(input_size, num_classes).to(device)
    # Define loss function and optimizer
    criterion = nn.BCELoss()  
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Training loop
    num_epochs = 100  
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(features_train)
        loss = criterion(outputs, labels_train)
        loss.backward()
        optimizer.step()

        # Threshold outputs to get binary predictions
        predicted = (outputs > 0.5).float()
        if epoch % 20 == 0:
            accuracy_train = accuracy_score(labels_train_cpu, predicted.cpu().numpy())
            precision_train = precision_score(labels_train_cpu, predicted.cpu().numpy(), average='micro')
            recall_train = recall_score(labels_train_cpu, predicted.cpu().numpy(), average='micro')
            f1 = f1_score(labels_train_cpu, predicted.cpu().numpy(), average='weighted')
            print('Epoch: {}. Trainset Accuracy: {}. Precision: {}. Recall: {}. F1 Score: {}.'.format(epoch, accuracy_train, precision_train, recall_train, f1))
    
    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(features_test)
        predicted = (outputs > 0.5).float()
        accuracy_test = accuracy_score(labels_test_cpu, predicted.cpu().numpy())
        precision_test = precision_score(labels_test_cpu, predicted.cpu().numpy(), average='micro')
        recall_test = recall_score(labels_test_cpu, predicted.cpu().numpy(), average='micro')
        f1 = f1_score(labels_test_cpu, predicted.cpu().numpy(), average='weighted')
        print('Testset Accuracy: {}. Precision: {}. Recall: {}. F1 Score: {}.'.format(accuracy_test, precision_test, recall_test, f1))

        labels = ['go', 'stop', 'warning']
        conf_matrix = multilabel_confusion_matrix(labels_test_cpu, predicted.cpu().numpy())
        for i, matrix in enumerate(conf_matrix):
            print(f"Confusion Matrix for Label {labels[i]}:")
            print(matrix)
            print()
            plt.imshow(matrix)
            plt.colorbar()
            plt.xlabel("Predicted")
            plt.xticks([0,1],[labels[i],"others"])
            plt.yticks([0,1],[labels[i],"others"])
            plt.ylabel("Real")
            plt.show()
    
    return predicted.cpu().numpy()