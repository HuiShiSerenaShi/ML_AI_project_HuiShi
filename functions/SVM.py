import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC


# SVM
# Multi target classification, fitting one classifier per target (one-vs-rest).
# Multi-output targets predicted across multiple predictors.
def multi_output_classifier_linear(kernel, max_iteration, SEED, X_train, labels_train, X_test):

    # Initialize SVM classifier
    svm_clf = MultiOutputClassifier(SVC(kernel=kernel, max_iter=max_iteration, probability=True, random_state=SEED)) 
    # Train the classifier
    svm_clf.fit(X_train, labels_train)
    # Predictions on the test set
    y_pred = svm_clf.predict(X_test)

    return y_pred

def multi_output_classifier_poly(kernel, degree, max_iteration, SEED, X_train, labels_train, X_test):

    # Initialize SVM classifier
    svm_clf = MultiOutputClassifier(SVC(kernel=kernel, degree=degree, max_iter=max_iteration, probability=True, random_state=SEED)) 
    # Train the classifier
    svm_clf.fit(X_train, labels_train)
    # Predictions on the test set
    y_pred = svm_clf.predict(X_test)

    return y_pred

def multi_output_classifier_rbf(kernel, gamma, max_iteration, SEED, X_train, labels_train, X_test):

    # Initialize SVM classifier
    svm_clf = MultiOutputClassifier(SVC(kernel=kernel, gamma=gamma, max_iter=max_iteration, probability=True, random_state=SEED)) 
    # Train the classifier
    svm_clf.fit(X_train, labels_train)
    # Predictions on the test set
    y_pred = svm_clf.predict(X_test)

    return y_pred

