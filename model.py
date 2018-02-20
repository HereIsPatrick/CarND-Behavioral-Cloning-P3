import csv
import cv2
from scipy.misc import imread
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Lambda
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint

images = []
steerings = []

random_seed = 666

lines = []
with open('../data/original/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    # Step . skip header
    _ = next(reader)

    for line in reader:
        (center, left, right, steering, throttle, brake, speed) = line
        steering = float(steering)

        correct = 0.2

        # add center image and steering info
        images.append(center)
        steerings.append(steering)

        # add left image and steering info
        images.append(left)
        steerings.append(steering+correct)

        # add left image and steering info
        images.append(right)
        steerings.append(steering-correct)

# shuffle it
images, steerings = shuffle(images, steerings, random_state=random_seed)

# split data
X_train, X_valid, y_train, y_valid = train_test_split(
    images, steerings, test_size=0.2, random_state=random_seed)

# check set size
assert len(X_train) == len(y_train)
assert len(X_valid) == len(y_valid)
print('train set : %d' % len(X_train))
print('valid set : %d' % len(X_valid))

# prepare data.
def prepare_data(image, steering, random_flip=False):
    abs_path = os.path.join('../data/original/', image.strip())
    img = imread(abs_path).astype(np.float32)

    # flip half of all images
    if random_flip and random.random() > 0.5:
        img = np.fliplr(img)
        steering = -steering

    return img, steering

# generator for batch
def batch_generator( images, steerings, batch_size=128, is_training=False):
    num_image = len(images)
    num_steering = len(steerings)
    assert num_image == num_steering

    while True:
        for offset in range(0, num_image, batch_size):
            X_batch = []
            y_batch = []

            stop = offset + batch_size
            image_b = images[offset:stop]
            steering_b = steerings[offset:stop]

            # prepare data for each batch
            for i in range(len(image_b)):
                image, steering = prepare_data(image_b[i], steering_b[i], random_flip=is_training)
                X_batch.append(image)
                y_batch.append(steering)

            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield X_batch, y_batch


def create_model():
    # Step . initialize with sequential.
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # Step . crop image, remove sky and hood
    model.add(Cropping2D(cropping=((60,25),(0,0))))

    # Step . use nVidia model
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    # Step . add dropout
    model.add(Dropout(0.5))

    # Step . full connect.
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model

X_train_size = len(X_train)
X_valid_size = len(X_valid)
batch_size = 128
nb_epoch = 3

def train_model(model, X_train, X_valid, y_train, y_valid):

    # Step . create generator
    train_generator = batch_generator(X_train, y_train, batch_size=batch_size, is_training=True)
    valid_generator = batch_generator(X_valid, y_valid, batch_size=batch_size, is_training=False)

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 save_best_only=True,
                                 mode='auto',period = 5)

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    model.fit_generator(train_generator,
                        X_train_size,
                        nb_epoch,
                        validation_data=valid_generator,
                        nb_val_samples=X_valid_size,
                        callbacks=[checkpoint],
                        verbose=1)
    model.save('model.h5')

def main():
    model = create_model()
    train_model(model, X_train, X_valid, y_train, y_valid)


if __name__=='__main__':
    main()
