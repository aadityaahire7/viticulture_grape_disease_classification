# viticulture_grape_disease_classification

Kaggle Dataset Link: https://www.kaggle.com/datasets/rm1000/grape-disease-dataset-original

Grape Disease Detection using Machine Learning
This repository contains the code and resources for detecting grape diseases using machine learning techniques. The dataset used in this project is sourced from Kaggle.

Project Overview
The goal of this project is to classify different types of grape diseases using images. We implement various machine learning and deep learning models to achieve high accuracy in classifying these diseases, helping grape farmers and agricultural experts detect and mitigate disease spread early.

Dataset
The dataset used in this project is sourced from Kaggle: Grape Disease Dataset (Original).

It contains images of grape leaves affected by various diseases, including:

Black rot
Esca (Black Measles)
Leaf Blight
Healthy grape leaves
The dataset is divided into training, validation, and test sets to ensure robust model performance.

Requirements
The following libraries are required to run the code:

Python 3.x
TensorFlow/Keras
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn

You can install the dependencies by running:


pip install -r requirements.txt

Usage

Clone the repository:

git clone https://github.com/yourusername/grape-disease-detection.git
cd grape-disease-detection
Download the dataset from Kaggle and place it in the data/ directory.

Run the Jupyter notebook grape_code.ipynb to train the model and evaluate its performance:

jupyter notebook grape_code.ipynb
The model will be trained and evaluated using accuracy, precision, recall, and F1 score metrics.

Model Architecture
The model architecture is based on a convolutional neural network (CNN) that is designed to extract features from the grape leaf images and classify them into the respective disease categories. The architecture consists of several convolutional, pooling, and dense layers with activation functions to build a robust model.


Future Improvements
Experimenting with different CNN architectures.
Hyperparameter tuning for better performance.
Deploying the model for real-time disease detection.
Contributing
Feel free to open issues or submit pull requests if you'd like to contribute!

License
This project is licensed under the MIT License. See the LICENSE file for more details.
