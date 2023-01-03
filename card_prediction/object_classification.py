import os

import cv2 as cv
import joblib
import numpy as np

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
TESTDIR = "card_test_data"
IMG_SIZE = 100

loaded_model = joblib.load(filename)
image_classification = {}
# for testing
# for category in LABELS:
path = os.path.join(TESTDIR)
for img in os.listdir(path):
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        new_img = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_shape = new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        predictions = loaded_model.predict(new_shape)
        image_lable = LABELS[np.argmax(predictions)]
        predected_imgs = image_classification.get(image_lable, [])
        predected_imgs.append('/' + img)
        image_classification.update({image_lable: predected_imgs})

    except Exception as e:
        pass
print('image_classification : ', image_classification)
