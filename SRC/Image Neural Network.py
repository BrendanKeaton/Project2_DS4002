#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:44:45 2023

@author: benlenox
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
import numpy as np
from keras.utils import to_categorical
import os


directory = os.getcwd()
directory = directory.replace('\\', '/')
#THIS PART IS THE CODE FOR PREPROCCESSING AND STANDARDIZING THE IMAGES

# Define the target size for resizing the images
target_size = (224, 224)  # Change this to your desired image size

# Function to preprocess a single image
def preprocess_image(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)
    
    # Resize the image to the target size
    img = img.resize(target_size, Image.LANCZOS)
    
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
X_train = [
directory + '/IMAGES/GreatWall_1.jpg',
directory + '/IMAGES/GreatWall_2.jpg',
directory + '/IMAGES/GreatWall_3.jpg',
directory + '/IMAGES/GreatWall_4.jpg',
directory + '/IMAGES/GreatWall_5.jpg',
directory + '/IMAGES/GreatWall_6.jpg',
directory + '/IMAGES/GreatWall_7.jpg',
directory + '/IMAGES/GreatWall_8.jpg',
directory + '/IMAGES/GreatWall_9.jpg',
directory + '/IMAGES/GreatWall_10.jpg',
directory + '/IMAGES/GreatWall_11.jpg',
directory + '/IMAGES/GreatWall_12.jpg',
directory + '/IMAGES/GreatWall_13.jpg',
directory + '/IMAGES/GreatWall_14.jpg',
directory + '/IMAGES/GreatWall_15.jpg',
directory + '/IMAGES/GreatWall_16.jpg',
directory + '/IMAGES/GreatWall_17.jpg',
directory + '/IMAGES/GreatWall_18.jpg',
directory + '/IMAGES/GreatWall_19.jpg',
directory + '/IMAGES/GreatWall_20.jpg',
directory + '/IMAGES/GreatWall_21.jpg',
directory + '/IMAGES/GreatWall_22.jpg',
directory + '/IMAGES/GreatWall_23.jpg',
directory + '/IMAGES/GreatWall_24.jpg',
directory + '/IMAGES/GreatWall_25.jpg',
directory + '/IMAGES/parth_1.jpg',
directory + '/IMAGES/parth_2.jpg',
directory + '/IMAGES/parth_3.jpg',
directory + '/IMAGES/parth_4.jpg',
directory + '/IMAGES/parth_5.jpg',
directory + '/IMAGES/parth_6.jpg',
directory + '/IMAGES/parth_7.jpg',
directory + '/IMAGES/parth_8.jpg',
directory + '/IMAGES/parth_9.jpg',
directory + '/IMAGES/parth_10.jpg',
directory + '/IMAGES/parth_11.jpg',
directory + '/IMAGES/parth_12.jpg',
directory + '/IMAGES/parth_13.jpg',
directory + '/IMAGES/parth_14.jpg',
directory + '/IMAGES/parth_15.jpg',
directory + '/IMAGES/parth_16.jpg',
directory + '/IMAGES/parth_17.jpg',
directory + '/IMAGES/parth_18.jpg',
directory + '/IMAGES/parth_19.jpg',
directory + '/IMAGES/parth_20.jpg',
directory + '/IMAGES/parth_21.jpg',
directory + '/IMAGES/parth_22.jpg',
directory + '/IMAGES/parth_23.jpg',
directory + '/IMAGES/parth_24.jpg',
directory + '/IMAGES/parth_25.jpg',
directory + '/IMAGES/Eif1.jpg',
directory + '/IMAGES/Eif2.jpg',
directory + '/IMAGES/Eif3.jpg',
directory + '/IMAGES/Eif4.jpg',
directory + '/IMAGES/Eif5.jpg',
directory + '/IMAGES/Eif6.jpg',
directory + '/IMAGES/Eif7.jpg',
directory + '/IMAGES/Eif8.jpg',
directory + '/IMAGES/Eif9.jpg',
directory + '/IMAGES/Eif10.jpg',
directory + '/IMAGES/Eif11.jpg',
directory + '/IMAGES/Eif12.jpg',
directory + '/IMAGES/Eif13.jpg',
directory + '/IMAGES/Eif14.jpg',
directory + '/IMAGES/Eif15.jpg',
directory + '/IMAGES/Eif16.jpg',
directory + '/IMAGES/Eif17.jpg',
directory + '/IMAGES/Eif18.jpg',
directory + '/IMAGES/Eif19.jpg',
directory + '/IMAGES/Eif20.jpg',
directory + '/IMAGES/Eif21.jpg',
directory + '/IMAGES/Eif22.jpg',
directory + '/IMAGES/Eif23.jpg',
directory + '/IMAGES/Eif24.jpg',
directory + '/IMAGES/Eif25.jpg',
directory + '/IMAGES/Unknown-12.jpeg',
directory + '/IMAGES/Unknown-11.jpeg',
directory + '/IMAGES/Unknown-10.jpeg',
directory + '/IMAGES/Unknown-9.jpeg',
directory + '/IMAGES/Unknown-8.jpeg',
directory + '/IMAGES/Unknown-7.jpeg',
directory + '/IMAGES/Unknown-6.jpeg',
directory + '/IMAGES/Unknown-5.jpeg',
directory + '/IMAGES/Unknown-4.jpeg',
directory + '/IMAGES/Unknown-3.jpeg',
directory + '/IMAGES/Unknown-2.jpeg',
directory + '/IMAGES/Unknown-1.jpeg',
directory + '/IMAGES/Unknown.jpeg',
directory + '/IMAGES/images.jpeg',
directory + '/IMAGES/images-1.jpeg',
directory + '/IMAGES/images-2.jpeg',
directory + '/IMAGES/images-3.jpeg',
directory + '/IMAGES/images-4.jpeg',
directory + '/IMAGES/images-5.jpeg',
directory + '/IMAGES/images-6.jpeg',
directory + '/IMAGES/images-7.jpeg',
directory + '/IMAGES/images-8.jpeg',
directory + '/IMAGES/images-9.jpeg',
directory + '/IMAGES/images-10.jpeg',
directory + '/IMAGES/images-11.jpeg',
directory + '/IMAGES/stonehenge (1).jpeg',
directory + '/IMAGES/stonehenge (2).jpeg',
directory + '/IMAGES/stonehenge (3).jpeg',
directory + '/IMAGES/stonehenge (4).jpeg',
directory + '/IMAGES/stonehenge (5).jpeg',
directory + '/IMAGES/stonehenge (6).jpeg',
directory + '/IMAGES/stonehenge (7).jpeg',
directory + '/IMAGES/stonehenge (8).jpeg',
directory + '/IMAGES/stonehenge (9).jpeg',
directory + '/IMAGES/stonehenge (10).jpeg',
directory + '/IMAGES/stonehenge (11).jpeg',
directory + '/IMAGES/stonehenge (12).jpeg',
directory + '/IMAGES/stonehenge (13).jpeg',
directory + '/IMAGES/stonehenge (14).jpeg',
directory + '/IMAGES/stonehenge (15).jpeg',
directory + '/IMAGES/stonehenge (16).jpeg',
directory + '/IMAGES/stonehenge (17).jpeg',
directory + '/IMAGES/stonehenge (18).jpeg',
directory + '/IMAGES/stonehenge (19).jpeg',
directory + '/IMAGES/stonehenge (20).jpeg',
directory + '/IMAGES/stonehenge (21).jpeg',
directory + '/IMAGES/stonehenge (22).jpeg',
directory + '/IMAGES/stonehenge (23).jpeg',
directory + '/IMAGES/stonehenge (24).jpeg',
directory + '/IMAGES/stonehenge (25).jpeg',
directory + '/IMAGES/stat1.jpg',
directory + '/IMAGES/stat2.jpg',
directory + '/IMAGES/stat3.jpg',
directory + '/IMAGES/stat4.jpg',
directory + '/IMAGES/stat5.jpg',
directory + '/IMAGES/stat6.jpg',
directory + '/IMAGES/stat7.jpg',
directory + '/IMAGES/stat8.jpg',
directory + '/IMAGES/stat9.jpg',
directory + '/IMAGES/stat10.jpg',
directory + '/IMAGES/stat11.jpg',
directory + '/IMAGES/stat12.jpg',
directory + '/IMAGES/stat13.jpg',
directory + '/IMAGES/stat14.jpg',
directory + '/IMAGES/stat15.jpg',
directory + '/IMAGES/stat16.jpg',
directory + '/IMAGES/stat17.jpg',
directory + '/IMAGES/stat18.jpg',
directory + '/IMAGES/stat19.jpg',
directory + '/IMAGES/stat20.jpg',
directory + '/IMAGES/stat21.jpg',
directory + '/IMAGES/stat22.jpg',
directory + '/IMAGES/stat23.jpg',
directory + '/IMAGES/stat24.jpg',
directory + '/IMAGES/stat25.jpg',
directory + '/IMAGES/MP1.jpg',
directory + '/IMAGES/MP2.jpg',
directory + '/IMAGES/MP3.jpg',
directory + '/IMAGES/MP4.jpg',
directory + '/IMAGES/MP5.jpg',
directory + '/IMAGES/MP6.jpg',
directory + '/IMAGES/MP7.jpg',
directory + '/IMAGES/MP8.jpg',
directory + '/IMAGES/MP9.jpg',
directory + '/IMAGES/MP10.jpg',
directory + '/IMAGES/MP11.jpg',
directory + '/IMAGES/MP12.jpg',
directory + '/IMAGES/MP13.jpg',
directory + '/IMAGES/MP14.jpg',
directory + '/IMAGES/MP15.jpg',
directory + '/IMAGES/MP16.jpg',
directory + '/IMAGES/MP17.jpg',
directory + '/IMAGES/MP18.jpg',
directory + '/IMAGES/MP19.jpg',
directory + '/IMAGES/MP20.jpg',
directory + '/IMAGES/MP21.jpg',
directory + '/IMAGES/MP22.jpg',
directory + '/IMAGES/MP23.jpg',
directory + '/IMAGES/MP24.jpg',
directory + '/IMAGES/MP25.jpg',
directory + '/IMAGES/TS1.jpg',
directory + '/IMAGES/TS2.jpg',
directory + '/IMAGES/TS3.jpg',
directory + '/IMAGES/TS4.jpg',
directory + '/IMAGES/TS5.jpg',
directory + '/IMAGES/TS6.jpg',
directory + '/IMAGES/TS7.jpg',
directory + '/IMAGES/TS8.jpg',
directory + '/IMAGES/TS9.jpg',
directory + '/IMAGES/TS10.jpg',
directory + '/IMAGES/TS11.jpg',
directory + '/IMAGES/TS12.jpg',
directory + '/IMAGES/TS13.jpg',
directory + '/IMAGES/TS14.jpg',
directory + '/IMAGES/TS15.jpg',
directory + '/IMAGES/TS16.jpg',
directory + '/IMAGES/TS17.jpg',
directory + '/IMAGES/TS18.jpg',
directory + '/IMAGES/TS19.jpg',
directory + '/IMAGES/TS20.jpg',
directory + '/IMAGES/TS21.jpg',
directory + '/IMAGES/TS22.jpg',
directory + '/IMAGES/TS23.jpg',
directory + '/IMAGES/TS24.jpg',
directory + '/IMAGES/TS25.jpg',
]


y_train = [
1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1, 1, # Great Wall
2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2, 2, # Parthenon
3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3, 3, # Eiffel Tower
4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4, 4, # Taj Mahal
5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5, 5, # Stonehenge
6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 	6, 6, # Statue of Liberty
7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7, 7, # MP
8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8,	8, 8 # Times Square
]

X_test = [
directory + '/IMAGES/GreatWall_26.jpg',
directory + '/IMAGES/GreatWall_27.jpg',
directory + '/IMAGES/GreatWall_28.jpg',
directory + '/IMAGES/GreatWall_29.jpg',
directory + '/IMAGES/GreatWall_30.jpg',
directory + '/IMAGES/parth_26.jpg',
directory + '/IMAGES/parth_27.jpg',
directory + '/IMAGES/parth_28.jpg',
directory + '/IMAGES/parth_29.jpg',
directory + '/IMAGES/parth_30.jpg',
directory + '/IMAGES/Eif26.jpg',
directory + '/IMAGES/Eif27.jpg',
directory + '/IMAGES/Eif28.jpg',
directory + '/IMAGES/Eif29.jpg',
directory + '/IMAGES/Eif30.jpg',
directory + '/IMAGES/Unknown-17.jpeg',
directory + '/IMAGES/Unknown-16.jpeg',
directory + '/IMAGES/Unknown-15.jpeg',
directory + '/IMAGES/Unknown-14.jpeg',
directory + '/IMAGES/Unknown-13.jpeg',
directory + '/IMAGES/stonehenge (26).jpeg',
directory + '/IMAGES/stonehenge (27).jpeg',
directory + '/IMAGES/stonehenge (28).jpeg',
directory + '/IMAGES/stonehenge (29).jpeg',
directory + '/IMAGES/stonehenge (30).jpeg',
directory + '/IMAGES/stat26.jpg',
directory + '/IMAGES/stat27.jpg',
directory + '/IMAGES/stat28.jpg',
directory + '/IMAGES/stat29.jpg',
directory + '/IMAGES/stat30.jpg',
directory + '/IMAGES/MP26.jpg',
directory + '/IMAGES/MP27.jpg',
directory + '/IMAGES/MP28.jpg',
directory + '/IMAGES/MP29.jpg',
directory + '/IMAGES/MP30.jpg',
directory + '/IMAGES/TS26.jpg',
directory + '/IMAGES/TS27.jpg',
directory + '/IMAGES/TS28.jpg',
directory + '/IMAGES/TS29.jpg',
directory + '/IMAGES/TS30.jpg',
]

y_test = [
1, 1, 1, 1, 1, # Great Wall
2, 2, 2, 2, 2, # Parthenon
3, 3, 3, 3, 3, # Eiffel Tower
4, 4, 4, 4, 4, # Taj Mahal
5, 5, 5, 5, 5, # Stonehenge
6, 6, 6, 6, 6, # Statue of Liberty
7, 7, 7, 7, 7, # MP
8, 8, 8, 8, 8] # Times Square

# 'preprocessed_data' now contains the preprocessed images

#THIS PART ACTUALLY BUILDS THE NEURAL NETWORK
 # PRe process the data
X_train = preprocess_dataset(X_train)
X_test = preprocess_dataset(X_test)


# Convert the encoded labels to one-hot encoded format
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Define your CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax') 
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Use categorical_crossentropy for one-hot encoded labels
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print(f'Test accuracy: {test_acc}')
print(model.summary())