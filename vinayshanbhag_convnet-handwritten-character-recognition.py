# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.activations import softmax, relu
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import RandomUniform, glorot_uniform
from keras.constraints import max_norm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn import preprocessing


%matplotlib inline
df = pd.read_csv("../input/devanagari-character-set/data.csv")
char_names = df.character.unique()  
rows =10;columns=5;
fig, ax = plt.subplots(rows,columns, figsize=(8,16))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        if columns*row+col < len(char_names):
            x = df[df.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)
            x = x.astype("float64")
            x/=255
            ax[row,col].imshow(x, cmap="binary")
            ax[row,col].set_title(char_names[columns*row+col].split("_")[-1])

            
plt.subplots_adjust(wspace=1, hspace=1)        
le = preprocessing.LabelEncoder()
label_vec = le.fit_transform(df.character)

X = df.iloc[:, :-1].values
X = X.reshape(X.shape[0], 32, 32, 1)
X = X.astype("float64")
X /= 255

y = to_categorical(label_vec)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("X_train",X_train.shape, "\ny_train",y_train.shape, "\n\nX_cv", X_cv.shape, "\ny_cv", y_cv.shape, "\n\nX_test", X_test.shape, "\ny_test",y_test.shape)
model = load_model("../input/pretrained-cnn-model/model.h5")
model.summary()
print("Accuracy:",model.evaluate(X_test, y_test, verbose=1)[1])
pred = model.predict(X_test, verbose=1)
print(classification_report(np.argmax(y_test, axis=1),np.argmax(pred, axis=1), target_names=[c.split("_")[-1] for c in le.classes_],digits=4))
def _low_confidence_idx(predicted, y, confidence=0.8):
    """Get indices of low confidence predictions"""
    a = []
    for i in range(predicted.shape[0]):
        if predicted[i][np.argsort(predicted[i])[-1]]<confidence:
            a.append(i)
    return a

def get_low_confidence_predictions(predictions, X, y, confidence=0.8):
    """get all info about ambiguous predictions"""
    idx = _low_confidence_idx(predictions, y, confidence=confidence)
    results = []
    for i in idx:
        result = dict()
        result["image"] = X[i].reshape(X.shape[1],X.shape[2])
        result["true_class"] = le.inverse_transform(np.argmax(y[i]))
        top2 = np.argsort(predictions[i])[-2:]
        predicted_classes = []
        for j in top2[::-1]:
            predicted_classes.append((le.inverse_transform(j), predictions[i][j]))
        result["predicted_classes"]=predicted_classes
        results.append(result)
    return results

def _high_confidence_idx(predicted, y, confidence=0.9):
    """Get indices of high confidence incorrect predictions"""
    a = []
    for i in range(predicted.shape[0]):
        if predicted[i][np.argsort(predicted[i])[-1]]>confidence and np.argmax(predicted[i]) != np.argmax(y[i]):
            a.append(i)
    return a

def get_high_confidence_errors(predictions, X, y, confidence=0.9):
    """get all info about ambiguous predictions"""
    idx = _high_confidence_idx(predictions, y, confidence=confidence)
    results = []
    for i in idx:
        result = dict()
        result["image"] = X[i].reshape(X.shape[1],X.shape[2])
        result["true_class"] = le.inverse_transform(np.argmax(y[i]))
        top2 = np.argsort(predictions[i])[-2:]
        predicted_classes = []
        for j in top2[::-1]:
            predicted_classes.append((le.inverse_transform(j), predictions[i][j]))
        result["predicted_classes"]=predicted_classes
        results.append(result)
    return results
"""Display source image, true and predicted (top 2) classes with softmax score"""
low_conf = get_low_confidence_predictions(pred, X_test, y_test, confidence=0.8)

rows =10;columns=8
fig, ax = plt.subplots(rows,columns, figsize=(13,24))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        s = ""
        for p in low_conf[row*columns+col]["predicted_classes"]:
            s += "{}:({:0.2f})\n".format(p[0].split("_")[-1], p[1])
        true_class = low_conf[row*columns+col]["true_class"].split("_")[-1]
        ax[row,col].set_title("True:{}\n\nPredicted:\n{}".format(true_class,s))
        ax[row,col].imshow(low_conf[row*columns+col]["image"], cmap="binary")

plt.subplots_adjust(wspace=2, hspace=2)          
"""Display source image, true and predicted (top 2) classes with softmax score"""
high_conf = get_high_confidence_errors(pred, X_test, y_test, confidence=0.9)
plt.imshow(high_conf[6]["image"], cmap="binary")
plt.axis("off");
rows =11;columns=5
fig, ax = plt.subplots(rows,columns, figsize=(10,35))
for row in range(rows):
    for col in range(columns):
        ax[row,col].set_axis_off()
        s = ""
        for p in high_conf[row*columns+col]["predicted_classes"]:
            s += "{}:({:0.2f})\n".format(p[0].split("_")[-1], p[1])
        true_class = high_conf[row*columns+col]["true_class"].split("_")[-1]
        ax[row,col].set_title("True:{}\n\nPredicted:\n{}".format(true_class,s))
        ax[row,col].imshow(high_conf[row*columns+col]["image"], cmap="binary")

plt.subplots_adjust(wspace=2, hspace=2)    
