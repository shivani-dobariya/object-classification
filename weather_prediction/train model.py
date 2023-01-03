import os

import cv2 as cv
import joblib
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

DATADIR = "wether_train_dataset"
LABELS = ["cloudy","shine","sunrise","rain"]
filename = 'whether_prediction.sav'

X_TRAIN = []
Y_TRAIN = []

IMG_SIZE = 100
for category in LABELS:
    path = os.path.join(DATADIR, category)
    class_num = LABELS.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv.imread(os.path.join(path, img))
            new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
            X_TRAIN.append(new_array)
            Y_TRAIN.append(class_num)
        except Exception as e:
            pass

X_TRAIN = np.array(X_TRAIN).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
Y_TRAIN = np.array(Y_TRAIN)

X_TRAIN = X_TRAIN / 255

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=X_TRAIN.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_TRAIN, Y_TRAIN, epochs=20, validation_split=0.1)
joblib.dump(model, filename)
