import os
import csv
from sklearn.model_selection import train_test_split
import cv2 as cv
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
import matplotlib.pyplot as plt

def initialise_model(og_img_shape):
    # CNN architecture by NVIDIA, ref: https://arxiv.org/abs/1604.07316

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape = og_img_shape))
    model.add(Conv2D(24, (5,5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Conv2D(36, (5,5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Conv2D(48, (5,5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1))

    model.summary()

    return model

def generator(samples, batch_size = 32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # read in images and steering angles for the batch
            for batch_sample in batch_samples:
                center_image = cv.imread((image_datapath+batch_sample[0]).replace(' ',''))
                left_image = cv.imread((image_datapath+batch_sample[1]).replace(' ',''))
                right_image = cv.imread((image_datapath+batch_sample[2]).replace(' ',''))
                correction = 0.2
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction # correction for treating left images as center
                right_angle = center_angle - correction # correction for treating right images as center
                images.extend([center_image[70:135, :],left_image[70:135, :],right_image[70:135, :]]) # cropping images
                angles.extend([center_angle,left_angle,right_angle])

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv.flip(image, 1)) # flipping image for data augmentation
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)

            yield sklearn.utils.shuffle(X_train, y_train)

def loss_tracking(error_tracking_object):
    print(error_tracking_object.history.keys())
    plt.plot(error_tracking_object.history['loss'])
    plt.plot(error_tracking_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

samples = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)

image_datapath = "../data/IMG/"

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

row, col, ch = 65, 320, 3
img_shape = row,col,ch

model = initialise_model(img_shape) 

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples*6), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)
# note, the total training samples are 6 times per epoch counting both original
# and flipped left, right and center images
model.save('model.h5')
error_tracking(history_object)

