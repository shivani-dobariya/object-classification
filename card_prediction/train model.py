import os

import cv2 as cv
import joblib
import numpy as np
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

DATADIR = "card_train_dataset"
LABELS = [
    'ace_of_clubs',
    'two_of_spades',
    'five_of_hearts',
    'queen_of_diamonds',
    'three_of_spades',
    'three_of_clubs',
    'joker',
    'jack_of_diamonds',
    'six_of_diamonds',
    'seven_of_clubs',
    'eight_of_spades',
    'ten_of_hearts',
    'five_of_spades',
    'ace_of_diamonds',
    'ace_of_hearts',
    'ten_of_diamonds',
    'ten_of_spades',
    'jack_of_spades',
    'six_of_clubs',
    'king_of_spades',
    'two_of_hearts',
    'nine_of_clubs',
    'four_of_clubs',
    'nine_of_spades',
    'ace_of_spades',
    'seven_of_diamonds',
    'king_of_diamonds',
    'five_of_clubs',
    'queen_of_hearts',
    'king_of_clubs',
    'seven_of_spades',
    'four_of_diamonds',
    'three_of_diamonds',
    'four_of_hearts',
    'jack_of_hearts',
    'king_of_hearts',
    'eight_of_hearts',
    'queen_of_clubs',
    'five_of_diamonds',
    'ten_of_clubs',
    'nine_of_hearts',
    'jack_of_clubs',
    'eight_of_clubs',
    'two_of_diamonds',
    'two_of_clubs',
    'seven_of_hearts',
    'six_of_spades',
    'three_of_hearts',
    'queen_of_spades',
    'nine_of_diamonds',
    'four_of_spades',
    'six_of_hearts',
    'eight_of_diamonds'
]
filename = 'card_prediction.sav'

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
model.add(Dense(len(LABELS)+2))
model.add(Activation("softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_TRAIN, Y_TRAIN, epochs=20, validation_split=0.1)
joblib.dump(model, filename)
