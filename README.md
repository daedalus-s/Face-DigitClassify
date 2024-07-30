# Face-DigitClassify

This project implements image classification models for face recognition and digit classification using TensorFlow and Keras.

## Project Overview

Face-DigitClassify contains two main components:

1. Face Recognition
2. Digit Classification

Both components use Convolutional Neural Networks (CNNs) for image classification tasks.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV (cv2)

You can install the required packages using:
pip install tensorflow keras numpy matplotlib opencv-python
## Face Recognition

The face recognition model is implemented in `face_recognition.py`. It uses a CNN architecture to classify faces into different categories.

### Dataset

The face dataset should be organized in the following structure:
face_dataset/
person1/
image1.jpg
image2.jpg
...
person2/
image1.jpg
image2.jpg
...
...
### Usage

To train and evaluate the face recognition model:

python
python face_recognition.py

### Digit Classification

The digit classification model is implemented in digit_classification.py. It uses a CNN to classify handwritten digits (0-9).
Dataset
The digit classification model uses the MNIST dataset, which is loaded automatically through Keras.
Usage
To train and evaluate the digit classification model:
python digit_classification.py

### Model Architecture
Both models use a similar CNN architecture:

### Convolutional layers with ReLU activation
Max pooling layers
Flatten layer
Dense layers with ReLU activation
Output layer with softmax activation

### The exact architecture can be found in the respective Python files.
### Results
The models' performance metrics (accuracy, loss) are printed during training and evaluation. Additionally, confusion matrices are generated to visualize the classification results.
Contributing
Contributions to improve the models or extend the project are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests.
