# Traffic Lights Classification Project
  The objective of this project is to implement and compare various methods for traffic lights
  classification which is a multi-label classification problem. The input are images of roads
  having traffic lights. One image can have multiple labels as there can be multiple traffic lights
  with various colors (i.e. green, red, yellow) in it. Thus, the expected output for each image is
  a three elements tuple indicating the existence of the lights of the three colors. For instance,
  if an image has both green and red lights in it, the correct classification result should be (1, 1, 0).

  The selected classification methods are SVM, KNN, NN.

# Dataset 
  The performance of the developed algorithms was benchmarked on the LISA Traffic Light Dataset.

  The dataset can be downloaded from 
  https://universe.roboflow.com/ithb-5ka4m/lisa-traffic-light-detection-8vuch/dataset/3/download , the format - multi-label classification should be specified before downloading.

  For more details, check the technical report.

# Algorithms & Experiments results
  basic modules are in the directory functions/

  algorithms combining the feature extraction methods and classification models are in the directories SVM_scripts/, KNN_scripts/, NN_scripts/. For implementation details, check the technical report.

  experiments results are included in the above mentioned scripts (.ipynb files). For a detailed analysis, check the technical report.