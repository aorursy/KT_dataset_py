import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import gc
import pandas as pd
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD 
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, AUC, BinaryAccuracy, Precision, Recall
import cv2
import math
from tensorflow.keras.regularizers import l1, l2, l1_l2
import shutil
from tqdm import tqdm
import os
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("../input/jpeg-melanoma-256x256/train.csv")
test = pd.read_csv("../input/jpeg-melanoma-256x256/test.csv")
train.loc[train["sex"].isnull(),["sex"]] = "male"
train.loc[train["age_approx"].isnull(),["age_approx"]] = 50
train.loc[train["anatom_site_general_challenge"].isnull(),["anatom_site_general_challenge"]] = "torso"
train["split"] = 0
train.loc[train["age_approx"]<=40,["split"]] = 1
train.loc[(train["age_approx"]>40) & (train["age_approx"]<=76),["split"]] = 2
train.loc[train["age_approx"]>76,["split"]] = 3
train["veil"] = 0
train["globuli"] = 0
test["veil"] = 0
test["globuli"] = 0
lower_red= np.array([153,112,131])
upper_red= np.array([215,161,170])
def get_blue_and_white_veil(im):
    im = im[30:SHAPE-30,30:SHAPE-30]
    mask = cv2.inRange(im, lower_red, upper_red)
    res = cv2.bitwise_and(im,im, mask= mask)
    thresh = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    thresh = cv2.Canny( res, 50, 70)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    jj=0
    for c in contours:
      if cv2.contourArea(c)>15:
        jj+=1

    blue_and_white_veil = 0  
    if jj>6:
      blue_and_white_veil=1
    return  blue_and_white_veil
lower_red2= np.array([0,0,0])
upper_red2= np.array([130,130,130])

def get_globuli2(im):
    img = im[30:SHAPE-30,30:SHAPE-30].copy()
    mask = cv2.inRange(img, lower_red2, upper_red2)
    kernel = np.ones((2,2),np.uint8)
    mask = cv2.dilate(mask,kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    jj=0
    tl_f = 0
    for c in contours:
      box = cv2.minAreaRect(c)
      ( tl, tr, br) = box
      if cv2.contourArea(c)<70 and  abs(tl[0]-tl_f)>15:
        jj+=1
      tl_f = tl[0]
    return jj
SHAPE=256
CROP_SIZE=15
centerXX=math.ceil((SHAPE-CROP_SIZE*2)/2)
centerYY=math.ceil((SHAPE-CROP_SIZE*2)/2)
D=115
TRAIN_DIR =  "../input/jpeg-melanoma-256x256/train/"
TEST_DIR = "../input/jpeg-melanoma-256x256/test/"
def constructing_features(DIR,DATAFRAME):
    for i, image_name in enumerate(tqdm(DATAFRAME["image_name"])):
        im = cv2.imread(DIR+image_name+".jpg")
        im = im[CROP_SIZE:SHAPE-CROP_SIZE,CROP_SIZE:SHAPE-CROP_SIZE]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        mask = np.full((im.shape[0],im.shape[0]), 0, dtype=np.uint8)
        cv2.circle(mask, (int(centerXX), int( centerYY)) , D , ( 255 , 0 , 0 ) , -1)
        im = cv2.bitwise_or(im, im, mask=mask)
        glob = get_globuli2(im)
        veil =  get_blue_and_white_veil(im)
        DATAFRAME.loc[i,"veil"] = veil
        DATAFRAME.loc[i,"globuli"] = glob
constructing_features(TRAIN_DIR,train)
train.head()
train_c = train.copy()
train_split = 0
train_val_split = 0

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=6)
for train_index, test_index in split.split(train_c,train_c["target"]):
    train_split = train_c.loc[train_index].copy()
    train_val_split = train_c.loc[test_index].copy()
    train_split.drop(["split"], axis=1, inplace=True)
    train_val_split.drop(["split"], axis=1, inplace=True)
train_x = train_split[["age_approx","veil","globuli","width","height"]]
train_y = train_split["target"]
val_x = train_val_split[["age_approx","veil","globuli","width","height"]]
val_y = train_val_split["target"]
METRICS = [
      TruePositives(name='tp'),
      FalsePositives(name='fp'),
      TrueNegatives(name='tn'),
      FalseNegatives(name='fn'), 
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc'),
]
CW = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_split["target"]),
                                                 train_split["target"])
clases = [0,1]
class_weights = dict(zip(clases,CW))
class_weights
table_input = Input(shape=(train_x.shape[1]))
d = Dense(300, activation="elu")(table_input)
d = BatchNormalization()(d)
d = Dropout(0.2)(d)
d = Dense(100, activation="elu")(d)
out = Dense(1, activation="sigmoid")(d)
model = Model(table_input,out)
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=METRICS)
model.summary()
hit = model.fit(x = train_x, y= train_y, validation_data=(val_x, val_y), epochs=10, class_weight=class_weights)
constructing_features(TEST_DIR,test)
test_x = test[["age_approx","veil","globuli","width","height"]]
predictions = model.predict(test_x)
test["target"] = predictions

test.head()
test_sub = test[["image_name","target"]]
test_sub.to_csv("submission33.csv", index=False, line_terminator="\n")
