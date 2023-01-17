import cv2

import os

import numpy as np

import glob

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder



from keras.datasets import mnist

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

from keras.models import Sequential, load_model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.regularizers import l2



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def merge_rects(rects, thresh):

    acceptedRects = list()

    rectsUsed = [False] * len(rects)

    

    # Pick one not processed rectangle from the list

    for supIdx, supVal in enumerate(rects):

        if not rectsUsed[supIdx]:

            currxMin = supVal[0]

            currxMax = supVal[0] + supVal[2]

            curryMin = supVal[1]

            curryMax = supVal[1] + supVal[3]

            

            # Calculate the middle point of it and mark it as used

            current_X_middle = (currxMin + currxMax) / 2

            current_Y_middle = (curryMin + curryMax) / 2

            rectsUsed[supIdx] = True

            

            # Pick one not processed rectangle from the list

            for subIdx, subVal in enumerate(rects[(supIdx + 1):], start=(supIdx + 1)):

                candxMin = subVal[0]

                candxMax = subVal[0] + subVal[2]

                candyMin = subVal[1]

                candyMax = subVal[1] + subVal[3]

                

                # Calculate the middle point of it

                candidate_X_middle = (candxMin + candxMax) / 2

                candidate_Y_middle = (candyMin + candyMax) / 2

                

                # Compute the euclidean distance between two computed middle points

                dist = np.abs(current_X_middle - candidate_X_middle)

                

                # If the distance is lower then some threshold, merge the rectangles and mark the rectangle as used

                if dist <= thresh and np.abs(current_Y_middle - candidate_Y_middle) < 55:

                    currxMax = max(currxMax, candxMax)

                    currxMin = min(currxMin, candxMin)

                    curryMin = min(curryMin, candyMin)

                    curryMax = max(curryMax, candyMax)

                    rectsUsed[subIdx] = True

            

            if (currxMax - currxMin) * (curryMax - curryMin) > 100:

                acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])

    return acceptedRects



def get_numbers(filename, prox=80, canny_thr=60):

    image = 255 - cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), dsize=(640, 480))

    canny_features = cv2.Canny(image, 10, canny_thr)

    contours, _ = cv2.findContours(canny_features, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    for contour in contours:

        (x, y, w, h) = cv2.boundingRect(contour)



        mean = np.mean(image[y:y + h, x:x + w])

        std = np.std(image[y:y + h, x:x + w])



        ret, canny_features[y:y + h, x:x + w] = cv2.threshold(image[y:y + h, x:x + w], mean + std / 2, 255, cv2.THRESH_BINARY)

        

    contours, _ = cv2.findContours(canny_features, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = list()



    for contour in contours:

        boxes.append(cv2.boundingRect(contour))

    

    acceptedRects = merge_rects(boxes, prox)

    

    result_list = list()

    for rect in acceptedRects:

        x, y, w, h = rect[0], rect[1], rect[2], rect[3]

        dilated_canny = cv2.dilate(canny_features[y:y + h, x:x + w], np.ones((7, 7)))

        contours, _ = cv2.findContours(dilated_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        digit_list = list()

        for contour in sorted_ctrs:

            (ix, iy, iw, ih) = cv2.boundingRect(contour)

            result_img = cv2.copyMakeBorder(dilated_canny[iy:iy + ih, ix:ix + iw], 10, 10, 10, 10, cv2.BORDER_CONSTANT)

            result_img = cv2.erode(result_img, np.ones((4, 4)))

            result_img = cv2.resize(result_img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

            digit_list.append(result_img)

        result_list.append(np.array(digit_list))

    return np.array(result_list)
!wget -O images.zip "https://www.dropbox.com/s/l45ikrgvzuv5scb/Camera%20Roll.zip?dl=0"

!unzip images.zip

!mv -v "Camera Roll" "test-images"
!ls test-images
model = Sequential()



model.add(Conv2D(8, kernel_size=5, padding="same", activation="relu", input_shape=(28, 28, 1)))

model.add(MaxPooling2D())



model.add(Conv2D(16, kernel_size=5, padding="same", activation="relu"))

model.add(MaxPooling2D())



model.add(Flatten())



model.add(Dense(128, kernel_regularizer=l2(5e-3), activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, kernel_regularizer=l2(5e-3), activation="softmax"))



model.summary()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train / 255

x_test = x_test / 255
one_hot_enc = OneHotEncoder(sparse=False)

one_hot_enc.fit(np.concatenate((y_train, y_test)).reshape(-1, 1))

y_train_transformed = one_hot_enc.transform(y_train.reshape(-1, 1))

y_test_transformed = one_hot_enc.transform(y_test.reshape(-1, 1))
model.compile(optimizer=Adam(lr=5e-4), loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(

    np.concatenate((x_train, x_test)),

    np.concatenate((y_train_transformed, y_test_transformed)),

    batch_size=100,

    epochs=20)
def recognize_numbers(numbers):

    for number in numbers:

        digits = list()

        for digit in number:

            inp = digit.reshape(1, 28, 28, 1) / 255

            digit_hat = model.predict(inp)

            digits.append(one_hot_enc.inverse_transform(digit_hat)[0][0].astype(int))

        print("".join([str(d) for d in digits]))
for image in glob.glob("test-images/*.jpg"):

    numbers = get_numbers(image, 300, 50)

    plt.imshow(cv2.imread(image, cv2.IMREAD_GRAYSCALE), cmap="gray")

    plt.show()

    recognize_numbers(numbers)