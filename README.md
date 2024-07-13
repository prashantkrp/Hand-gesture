Hand Gesture Recognition:
This project implements a hand gesture recognition system using TensorFlow, Keras, OpenCV, and MediaPipe. The system can classify different hand gestures in real-time using a webcam.

Project Structure:
The project includes scripts for training a CNN model and for testing the model with real-time webcam input.

Datasets:
The dataset used for training is named leapGestRecog, which contains images of different hand gestures.

Libraries Used For Training:
TensorFlow
Keras
NumPy
Pillow (PIL)

For Final Testing:
OpenCV
MediaPipe
TensorFlow
NumPy
Pillow (PIL)


File Descriptions:

trained_model.py: Script to train the hand gesture recognition model. It preprocesses the dataset, builds and trains the CNN model, and saves the trained model.
test_model.py: Script to test the hand gesture recognition model in real-time using a webcam.
hand_gesture_recognition_model.h5: The file where the trained hand gesture recognition model is saved.
How It Works
Data Preprocessing: The dataset is augmented and preprocessed using ImageDataGenerator.
Model Building: A Convolutional Neural Network (CNN) is built and trained on the dataset.
Model Training: The model is trained on the dataset of hand gesture images and saved.
Real-Time Prediction: The trained model is used to recognize hand gestures in real-time using a webcam.
Acknowledgements
TensorFlow: An open-source platform for machine learning.
Keras: A deep learning API written in Python, running on top of TensorFlow.
OpenCV: An open-source computer vision and machine learning software library.
MediaPipe: A framework for building multimodal (e.g., video, audio, time-series) applied machine learning pipelines.
