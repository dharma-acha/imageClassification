## Image Classification with VGG 13 model
*Interactive web application using Streamlit that leverages the VGG-13 model to classify images from the CIFAR-10 dataset. The application allows users to upload images and receive real-time predictions along with visual explanations of the model's decisions*

![result](https://github.com/dharma-acha/imageClassification/assets/100557655/690110d8-3fd7-4572-93d3-21e4058af9d9)


### About

This project involves building and evaluating an image classification model using the VGG-13 architecture on the CIFAR-10 dataset. The primary goal is to accurately classify images into one of the ten classes provided by CIFAR-10, which include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

### Objectives

* **Implement VGG-13 Model:** Utilize the VGG-13 architecture, known for its deep convolutional layers, to build a robust image classifier.

* **Data Preprocessing:** Prepare the CIFAR-10 dataset for training and testing, including normalization and data augmentation to improve model performance.

* **Model Training:** Train the VGG-13 model on the training set of CIFAR-10 and optimize it using techniques like learning rate scheduling and early stopping.

* **Model Evaluation:** Assess the model's performance on the test set using metrics such as accuracy, precision, recall, and F1-score

### Dataset Overview

The CIFAR-10 dataset is a widely used benchmark in machine learning and computer vision. It consists of 60,000 color images, each of size 32x32 pixels, divided into 10 different classes with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. The classes represent different objects such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks, providing a diverse range of visual categories for evaluating classification algorithms.

### Technologies used

1. Scikit-Learn
2. Matplotlib
3. Seaborn
4. Google colab
5. Pandas
6. Numpy
7. Python
8. Pytorch
9. Streamlit

### Getting started

#### Installation steps

1. clone the repo
2. run the Google colab
3. provide the correct path of 'pth' file in app.py file
3. run the app.py file using command 'streamlit run <filename.py>'
4. connect to T4 GPU for faster training process

See the report for more details

### License

Free to use, no restrictions

### Contact

If you have any questions or comments, feel free to contact me on [Email](mailto:achadharma333@gmail.com)


