#Importing Necessary Libraries.

from PIL import Image

import numpy as np

import os

import cv2

import keras

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

data=[]

labels=[]

men=os.listdir("../input/data/men/")

print(men)

for a in men:

    try:

        image=cv2.imread("../input/data/men/"+a)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((200, 200))

        data.append(np.array(size_image))

        labels.append(0)

    except AttributeError:

        print("")

women=os.listdir("../input/data/women/")

for b in women:

    try:

        image=cv2.imread("../input/data/women/"+b)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((200, 200))

        data.append(np.array(size_image))

        labels.append(1)

    except AttributeError:

        print("")



print("data Prepearaion finished")
Cells=np.array(data)

labels=np.array(labels)

print(Cells.shape)

print(labels.shape)
np.save("Cells",Cells)

np.save("labels",labels)
Cells=np.load("Cells.npy")

labels=np.load("labels.npy")
s=np.arange(Cells.shape[0])

np.random.shuffle(s)

Cells=Cells[s]

labels=labels[s]
num_classes=len(np.unique(labels))

len_data=len(Cells)
(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]

x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by divinding 255.

x_test = x_test.astype('float32')/255

train_len=len(x_train)

test_len=len(x_test)
(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

print(y_train)

print(y_test)

print(x_train.shape)

print(y_train.shape)
#Doing One hot encoding as classifier has multiple classes

y_train=keras.utils.to_categorical(y_train,2)

y_test=keras.utils.to_categorical(y_test,2)
#creating sequential model

model=Sequential()

model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(200,200,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 

model.summary()
# compile the model with loss as categorical_crossentropy and using adam optimizer you can test result by trying RMSProp as well as Momentum

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fit the model with min batch size as 50[can tune batch size to some factor of 2^power ] 

model.fit(x_train,y_train,batch_size=128,epochs=30)
# Save the model weights:

from keras.models import load_model

model.save('men_women.h5')
from keras.models import load_model

import matplotlib.pyplot as plt

from PIL import Image

from PIL import Image

import numpy as np

import os

import cv2

def convert_to_array(img):

    im = cv2.imread(img)

    cv_rgb =cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    plt.imshow(cv_rgb)

    plt.show()

    img_ = Image.fromarray(im, 'RGB')

    image = img_.resize((200, 200))

    

    return np.array(image)

def get_cell_name(label):

    if label==0:

        return "men"

    if label==1:

        return "women"

def predict_cell(file):

    model = load_model('men_women.h5')

    print("Predicting Type of people Image.................................")

    ar=convert_to_array(file)

    ar=ar/255

    label=1

    a=[]

    a.append(ar)

    a=np.array(a)

    score=model.predict(a,verbose=1)

    print(score)

    label_index=np.argmax(score)

    print(label_index)

    acc=np.max(score)

    Cell=get_cell_name(label_index)

    return Cell,"The people Cell is a "+Cell+" with accuracy =    "+str(acc)

predict_cell('../input/data/men/00000001.jpg')

predict_cell('../input/data/women/00000002.jpg')

#Check the accuracy on Test data:

accuracy = model.evaluate(x_test, y_test, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
