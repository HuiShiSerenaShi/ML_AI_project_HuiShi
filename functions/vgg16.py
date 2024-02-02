import torch
import os
import numpy as np
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG16_Weights

# Initializing compute device (use GPU if available).
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# extract features using vgg16
def extract_features_and_labels(mean, std, train_folder_path, test_folder_path, train_labels,test_labels):
 
    normalization_std = std
    normalization_mean = mean
    
    loader  = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomResizedCrop(224),
                                transforms.Normalize(mean=normalization_mean,
                                                    std=normalization_std)])

    # Initialize the model.
    model = models.vgg16(weights=VGG16_Weights).features.to(device) 

    def image_loader(image_name): 
        image = Image.open(image_name).convert('RGB')
        # Fake batch dimension required to fit network's input dimensions.
        image = loader(image).unsqueeze(0)
        return image.to(device)

    def extract_features(model, image_path):
        image = image_loader(image_path)
        features = model(image)
        return features.data.detach().cpu().numpy().flatten()

    features_train = []
    labels_train = []
    # Extract features for each image in the training set
    for idx, row in train_labels.iterrows():
        image_path = os.path.join(train_folder_path, row['filename'])
        label = row.iloc[1:].values.astype(int)
        # Extract features
        features = extract_features(model, image_path)
        features_train.append(features)
        labels_train.append(label)

    features_train = np.array(features_train)
    labels_train = np.array(labels_train)

    features_test = []
    labels_test = []
    # Extract features for each image in the test set
    for idx, row in test_labels.iterrows():
        image_path = os.path.join(test_folder_path, row['filename'])
        label = row.iloc[1:].values.astype(int)
        # Extract features
        features = extract_features(model, image_path)
        features_test.append(features)
        labels_test.append(label)

    features_test = np.array(features_test)
    labels_test = np.array(labels_test)

    return features_train, labels_train, features_test, labels_test