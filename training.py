# Import required libraries for this section

import numpy as np
import matplotlib.pyplot as plt
import math
import cv2                     # OpenCV library for computer vision
from PIL import Image
import time 
from utils import *

# Import deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))



## TODO: Specify a CNN architecture
# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)

model = Sequential()
model.add(Convolution2D(8, (3,3), input_shape = X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.3))
model.add(Dense(30))

# Summarize the model
model.summary()

## TODO: Compile the model

epochs = 50
learning_rate = 0.01
decay_rate = learning_rate / epochs

#momentum = 0.8
#sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

## TODO: Train the model
hist = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=30, verbose=1)

## TODO: Save the model as model.h5
model.save('my_model.h5')
