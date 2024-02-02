import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_explained_variance(X_train_std, RAND_STATE):
    # Apply PCA without specifying the number of components to keep, in order to understand how many PCs are sufficient.
    pca = PCA(random_state=RAND_STATE) 
    pca.fit(X_train_std) 

    # Plot the explained variance ratio in a cumulative fashion, in order to visualize the cumulative variance plot.
    plt.plot(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained variance by number of components')
    plt.ylabel('Cumulative explained variance')
    plt.xlabel('Nr. of principal components')
    plt.show()

def pca(n_components, RAND_STATE, X_train_std, X_test_std):
    # Use the number of components needed to explain more than 80% of the variance in order to create a new PCA object and
    # perform the final dimensionality reduction on the data.
    pca = PCA(n_components=n_components, random_state=RAND_STATE)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    return X_train_pca, X_test_pca