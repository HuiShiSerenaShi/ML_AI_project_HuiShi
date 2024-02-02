import numpy as np
from sklearn.preprocessing import StandardScaler

# Standardize features by removing the mean and scaling to unit variance.
def standard_scaler(features_train, features_test):

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(features_train)
    X_test_std = scaler.transform(features_test) 

    return X_train_std, X_test_std