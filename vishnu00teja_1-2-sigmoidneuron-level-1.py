import numpy as np

import pandas as pd

import sys

import pickle

from PIL import Image, ImageFilter

from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os







np.random.seed(100)

LEVEL = 'level_1'
print(os.listdir("../input/contest-1-2/"))
def read_all(folder_path, key_prefix=""):

    '''

    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.

    '''

    print("Reading:")

    images = {}

    files = os.listdir(folder_path)

    for i, file_name in tqdm_notebook(enumerate(files), total=len(files)):

        file_path = os.path.join(folder_path, file_name)

        image_index = key_prefix + file_name[:-4]

        image = Image.open(file_path)

        image = image.convert("L")

        images[image_index] = np.array(image.copy()).flatten()

        image.close()

    return images

    
languages = ['ta', 'hi', 'en']
# inputting datasets and calling 'read_all()' function

images_train = read_all("../input/contest-1-2/level_1_train/"+LEVEL+"/"+"background", key_prefix='bgr_') # change the path

for language in languages:

    images_train.update(read_all("../input/contest-1-2/level_1_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))

print(len(images_train))



images_test = read_all("../input/contest-1-2/level_1_test/kaggle_"+LEVEL, key_prefix='') # change the path

print(len(images_test))
list(images_test.keys())[:5]
X_train = []

Y_train = []

for key,value in images_train.items():

    X_train.append(value)

    if (key[:4] == 'bgr_'):

        Y_train.append(0)

    else:

        Y_train.append(1)

        

ID_test = []

X_test = []

for key,value in images_test.items():

    X_test.append(value)

    ID_test.append(int(key))

    

X_train = np.asarray(X_train)

Y_train = np.asarray(Y_train)

X_test = np.asarray(X_test)



print(X_train.shape, Y_train.shape)

print(X_test.shape)
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)

X_scaled_test = scaler.transform(X_test)
print(X_scaled_train.shape, Y_train.shape)
class SigmoidNeuron:

    

    def __init__(self):

        self.w = None

        self.b = None

        

    def perceptron(self, x):

        return np.dot(x, self.w.T) + self.b

    

    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))

    

    def mse_grad_w(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred) * x

    

    def mse_grad_b(self, x , y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * y_pred * (1 - y_pred)

    

    def ce_grad_w(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y) * x

    

    def ce_grad_b(self, x, y):

        y_pred = self.sigmoid(self.perceptron(x))

        return (y_pred - y)

    

    def predict(self, X):

        Y_pred = []

        for x in X:

            y_pred = self.sigmoid(self.perceptron(x))

            Y_pred.append(y_pred)

        return np.asarray(Y_pred)

    

    def fit(self, X, Y, epochs = 1, learning_rate = 1, loss_fn = 'mse', initialize = True, display_loss = False):

        

        if initialize:

            self.w = np.random.randn(1, X.shape[1])

            self.b = 0

            

        if display_loss:

            loss = {}

            

        for i in tqdm_notebook(range(epochs), total = epochs, unit = "epoch"):

            dw = 0

            db = 0

            for x,y in zip(X, Y):

                if loss_fn == 'mse':

                    dw += self.mse_grad_w(x, y)

                    db += self.mse_grad_b(x, y)

                elif loss_fn == 'ce':

                    dw += self.ce_grad_w(x, y)

                    db += self.ce_grad_b(x, y)

            self.w -= learning_rate * dw

            self.b -= learning_rate * db

            

            if display_loss:

                Y_pred = self.sigmoid(self.perceptron(X))

                if loss_fn == 'mse':

                    loss[i] = mean_squared_error(Y, Y_pred)

                elif loss_fn == 'ce':

                    loss[i] = log_loss(Y, Y_pred)

                    

        if display_loss:

            plt.plot(np.array(list(loss.values())).astype(float))

            plt.xlabel('Epochs')

            if loss_fn == 'mse':

                plt.ylabel('Mean Squared Error')

            elif loss_fn == 'ce':

                plt.ylabel('Cross Entropy error')

            plt.show()

    

    
sn_mse = SigmoidNeuron()
sn_mse.fit(X_scaled_train, Y_train, epochs = 10000, learning_rate = 0.015, loss_fn = 'mse', initialize = True, display_loss = True)
sn_ce = SigmoidNeuron()
sn_ce.fit(X_scaled_train, Y_train, epochs = 10000, learning_rate = 0.015, loss_fn = 'ce', initialize = True, display_loss = True)
def print_accuracy(sn):

    Y_pred_train = sn.predict(X_scaled_train)

    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype('int').ravel()

    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)

    print(accuracy_train)

    
print_accuracy(sn_mse)

print_accuracy(sn_ce)
Y_pred_test = sn_ce.predict(X_scaled_test)

Y_pred_binarised_test = (Y_pred_test >= 0.5).astype("int").ravel()



submission = {}

submission['ImageId'] = ID_test

submission['Class'] = Y_pred_binarised_test



submission = pd.DataFrame(submission)

submission = submission[['ImageId', 'Class']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)