# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install git+https://github.com/rcmalli/keras-vggface.git
from collections import defaultdict
from glob import glob
from random import choice, sample

import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from tqdm import tqdm_notebook


train_file_path = "../input/kinship-recognition/train_relationships.csv"
train_folders_path = "../input/kinship-recognition/train/"
val_famillies_list = ["F07", "F08", "F09"]
all_images = glob(train_folders_path + "*/*/*.jpg")

def get_train_val(val_famillies):
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]
    
    return train, val, train_person_to_images_map, val_person_to_images_map
def read_img(path):
    img = cv2.imread(path)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels
n_val_famillies_list = len(val_famillies_list)
def baseline_model_9():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))
    base_model = VGGFace(model='resnet50', include_top=False)
    for x in base_model.layers[:-3]:
        x.trainable = True
    x1 = base_model(input_1)
    x2 = base_model(input_2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])
    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x5 = Multiply()([x1, x2])
    x = Concatenate(axis=-1)([x3, x4, x5])
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model([input_1, input_2], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()
    return model

for i in tqdm_notebook(range(n_val_famillies_list)):
    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])
    file_path = f"model9_{i}.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau]
    model9 = baseline_model_9()
    #model9.load_weights(file_path)
    #model9.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
    #                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=25, verbose=1,
    #                    workers = 4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)
model9_0 = baseline_model_9()
model9_0.load_weights("../input/models-weight-config/model9_0.h5")
model9_1 = baseline_model_9()
model9_1.load_weights("../input/models-weight-config/model9_1.h5")
model9_2 = baseline_model_9()
model9_2.load_weights("../input/models-weight-config/model9_2.h5")
def baseline_model_10():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))
    base_model = VGGFace(model='resnet50', include_top=False)
    for layer in base_model.layers[:-3]:
        layer.trainable = True
    x1 = base_model(input_1)
    x2 = base_model(input_2)
    merged_add = Add()([x1, x2])
    merged_sub = Subtract()([x1,x2])
    merged_add = Conv2D(100 , [1,1] )(merged_add)
    merged_sub = Conv2D(100 , [1,1] )(merged_sub)
    merged = Concatenate(axis=-1)([merged_add, merged_sub])
    merged = Flatten()(merged)
    merged = Dense(100, activation="relu")(merged)
    merged = Dropout(0.3)(merged)
    merged = Dense(25, activation="relu")(merged)
    merged = Dropout(0.3)(merged)
    out = Dense(1, activation="sigmoid")(merged)
    model = Model([input_1, input_2], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()
    return model

for i in tqdm_notebook(range(n_val_famillies_list)):
    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])
    file_path = f"model10_{i}.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau]
    #model10 = baseline_model_10()
    #model10.load_weights(file_path)
    #model10.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
    #                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=1,
    #                    workers = 4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

model10_0 = baseline_model_10()
model10_0.load_weights("../input/models-weight-config/model10_0.h5")
model10_1 = baseline_model_10()
model10_1.load_weights("../input/models-weight-config/model10_1.h5")
model10_2 = baseline_model_10()
model10_2.load_weights("../input/models-weight-config/model10_2.h5")
test_path = "../input/kinship-recognition/test/"

def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

submission = pd.read_csv('../input/kinship-recognition/sample_submission.csv')
models = [model9_0, model9_1, model9_2, model10_0, model10_1, model10_2]
preds_for_sub = np.zeros(submission.shape[0])

ensemble_predictions = [[] for i in range(len(models))]
for batch in tqdm_notebook(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])
    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])
    for i in range(len(models)):
        pred = models[i].predict([X1, X2]).ravel().tolist()
        ensemble_predictions[i] += pred
predictions = [(sum(x)/len(x)) for x in zip(*ensemble_predictions)]

submission['is_related'] = predictions
submission.to_csv("vgg_face_cv.csv", index=False)