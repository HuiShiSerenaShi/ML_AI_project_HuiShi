import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from matplotlib import pyplot as plt

# print Accuracy, Precision, Recall, F1 Score
# Compute and plot a confusion matrix for each class (one-vs-rest)
def cal_metrics(labels_test, predictions):
    
    accuracy = accuracy_score(labels_test, predictions) 
    precision = precision_score(labels_test, predictions, average='micro')
    recall = recall_score(labels_test, predictions, average='micro')
    f1 = f1_score(labels_test, predictions, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    labels = ['go', 'stop', 'warning']
    conf_matrix = multilabel_confusion_matrix(labels_test, predictions)
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

