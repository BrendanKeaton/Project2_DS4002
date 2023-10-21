#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:44:45 2023

@author: benlenox
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import numpy as np

#THIS PART IS THE CODE FOR PREPROCCESSING AND STANDARDIZING THE IMAGES

# Define the target size for resizing the images
target_size = (224, 224)  # Change this to your desired image size

# Function to preprocess a single image
def preprocess_image(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)
    
    # Resize the image to the target size
    img = img.resize(target_size, Image.ANTIALIAS)
    
    # Convert the image to a NumPy array
    img_array = np.array(img)
    
    # Normalize the pixel values (assuming 0-255 range)
    img_array = img_array / 255.0
    
    return img_array

# Function to preprocess a list of image paths
def preprocess_dataset(image_paths):
    # Initialize an empty list to store the preprocessed images
    preprocessed_images = []
    
    for image_path in image_paths:
        img_array = preprocess_image(image_path)
        preprocessed_images.append(img_array)
    
    # Convert the list to a NumPy array
    preprocessed_images = np.array(preprocessed_images)
    
    return preprocessed_images

# Example usage:
# Replace 'image_paths' with a list of file paths to your dataset
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
preprocessed_data = preprocess_dataset(image_paths)

# 'preprocessed_data' now contains the preprocessed images

#THIS PART ACTUALLY BUILDS THE NEURAL NETWORK

# Load and preprocess the dataset (replace with your own dataset loading code)
# X_train, y_train = ...
# X_test, y_test = ...

# Preprocess the data (e.g., resizing and normalizing)
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Define your CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')