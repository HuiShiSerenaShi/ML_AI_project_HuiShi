import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

# KNN
# Multi target classification, fitting one classifier per target (one-vs-rest).
# Multi-output targets predicted across multiple predictors.
def multi_output_classifier(n_neighbors, X_train, labels_train, X_test):

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    multioutput_KNNclassifier = MultiOutputClassifier(knn_classifier)
    multioutput_KNNclassifier.fit(X_train, labels_train)
    predictions = multioutput_KNNclassifier.predict(X_test)

    return predictions



