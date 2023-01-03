import os

import cv2 as cv
import joblib
import numpy as np

LABELS = ["cloudy","shine","sunrise","rain"]
filename = 'whether_prediction.sav'
TESTDIR = "weather_test_data"
IMG_SIZE = 100

loaded_model = joblib.load(filename)
image_classification = {}
# for testing
for img in os.listdir(TESTDIR):
    try:
        img_array = cv.imread(os.path.join(TESTDIR, img))
        new_img = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_shape = new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        predictions = loaded_model.predict(new_shape)
        image_lable = LABELS[np.argmax(predictions)]
        predected_imgs = image_classification.get(image_lable, [])
        predected_imgs.append(img)
        image_classification.update({image_lable: predected_imgs})
    except Exception as e:
        pass
print('image_classification : ' ,image_classification)
