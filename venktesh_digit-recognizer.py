import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import pandas as pd

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

import keras

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

train = pd.read_csv('../input/train.csv')

train.head()
# trace1 = go.Scatter(

#     x=list(range(784)),

#     y= cum_var_exp,

#     mode='lines+markers',

#     name="'Cumulative Explained Variance'",

# #     hoverinfo= cum_var_exp,

#     line=dict(

#         shape='spline',

#         color = 'goldenrod'

#     )

# )

# trace2 = go.Scatter(

#     x=list(range(784)),

#     y= var_exp,

#     mode='lines+markers',

#     name="'Individual Explained Variance'",

# #     hoverinfo= var_exp,

#     line=dict(

#         shape='linear',

#         color = 'black'

#     )

# )

# fig = tls.make_subplots(insets=[{'cell': (1,1), 'l': 0.7, 'b': 0.5}],

#                           print_grid=True)



# fig.append_trace(trace1, 1, 1)

# fig.append_trace(trace2,1,1)

# fig.layout.title = 'Explained Variance plots - Full and Zoomed-in'

# fig.layout.xaxis = dict(range=[0, 80], title = 'Feature columns')

# fig.layout.yaxis = dict(range=[0, 60], title = 'Explained Variance')
target = train['label']

train = train.drop('label', axis=1)

from sklearn.decomposition import PCA

n_components = 60

pca = PCA(n_components=n_components).fit(train.values)



eigenvalues = pca.components_.reshape(n_components, 28, 28)



# Extracting the PCA components ( eignevalues )

eigenvalues = pca.components_
n_row = 5

n_col = 12



# Plot the first 28 eignenvalues

plt.figure(figsize=(13,12))

for i in list(range(n_row * n_col)):

    offset =0

    plt.subplot(n_row, n_col, i + 1)

    plt.imshow(eigenvalues[i].reshape(28,28), cmap='jet')

    title_text = 'Eigenvalue ' + str(i + 1)

    plt.title(title_text, size=6.5)

    plt.xticks(())

    plt.yticks(())

plt.show()
plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
# pca = PCA(n_components= 0.8388)

# pca.fit(X_std)

# X_5d = pca.transform(X_std)

# Target = target
# trace0 = go.Scatter(

#     x = X_5d[:,0],

#     y = X_5d[:,1],

# #     name = Target,

# #     hoveron = Target,

#     mode = 'markers',

#     text = Target,

#     showlegend = False,

#     marker = dict(

#         size = 8,

#         color = Target,

#         colorscale ='Jet',

#         showscale = False,

#         line = dict(

#             width = 2,

#             color = 'rgb(255, 255, 255)'

#         ),

#         opacity = 0.8

#     )

# )

# data = [trace0]



# layout = go.Layout(

#     title= 'Principal Component Analysis (PCA)',

#     hovermode= 'closest',

#     xaxis= dict(

#          title= 'First Principal Component',

#         ticklen= 5,

#         zeroline= False,

#         gridwidth= 2,

#     ),

#     yaxis=dict(

#         title= 'Second Principal Component',

#         ticklen= 5,

#         gridwidth= 2,

#     ),

#     showlegend= True

# )





# fig = dict(data=data, layout=layout)

# py.iplot(fig, filename='styled-scatter')
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

del train

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

X = (train.drop(labels="label",axis=1))

Y = (train["label"])

X_train = X / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y, num_classes = 10)

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size  = 0.666666, random_state  = 0)

X_test.shape
# from sklearn.neighbors import KNeighborsClassifier

# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(include_bias = False)

# xtrain = pca.fit_transform(X_train)

# xtest = pca.transform(X_test)

# xtrainpoly = poly.fit_transform(xtrain)

# xtestpoly = poly.transform(xtest)

# # clf = KNeighborsClassifier(n_neighbors=16)





# # clf = clf.fit(xtrainpoly, y_train)

# # output_label = clf.predict(xtestpoly)
# from keras.models import Sequential

# from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPool2D

# from keras import optimizers

# from keras.callbacks import LearningRateScheduler

# nets = 3

# model = [0] *nets

# for j in range(3):

#     model[j] = Sequential()

#     model[j].add(Conv2D(24,kernel_size=5,padding="same", activation="relu"))

#     model[j].add(MaxPooling2D())

#     model[j].add(Dropout(0.4))

#     if j > 0:

#         model[j].add(Conv2D(48,kernel_size=5,padding="same", activation="relu"))

#         model[j].add(MaxPooling2D())

#         model[j].add(Dropout(0.4))

#     if j > 1:

#         model[j].add(Conv2D(64,kernel_size=5,padding="same", activation="relu"))

#         model[j].add(MaxPooling2D())

#         model[j].add(Dropout(0.4))

#     model[j].add(Flatten())

#     model[j].add(Dense(256, activation="relu"))

#     model[j].add(Dropout(0.4))

#     model[j].add(Dense(10, activation="softmax"))

#     model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# history = [0] * nets

# names = ["(C-P)x1","(C-P)x2","(C-P)x3"]

# epochs = 20

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+epochs))



# for j in range(nets):

#     history[j] = model[j].fit(X_train,y_train, batch_size=80, epochs = epochs, 

#         validation_data = (X_test,y_test), callbacks=[annealer], verbose=0)

#     print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

#         names[j],epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))

# plt.figure(figsize=(15,5))

# for i in range(nets):

#     plt.plot(history[i].history['val_acc'])

# plt.title('model accuracy')

# plt.ylabel('accuracy')

# plt.xlabel('epoch')

# plt.legend(names, loc='upper left')

# axes = plt.gca()

# axes.set_ylim([0.98,1])

# plt.show()
from keras.models import Sequential

from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, MaxPool2D, BatchNormalization

from keras import optimizers

from keras.callbacks import LearningRateScheduler

model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** (x+25))

model.fit(X_train,y_train, batch_size=64, epochs = 25, callbacks=[annealer], verbose=0)

output_label = model.predict(X_test)
output = pd.DataFrame(np.argmax(output_label, axis=1),columns = ['Label'])

output.reset_index(inplace=True)

output['index'] = output['index'] + 1

output.rename(columns={'index': 'ImageId'}, inplace=True)

output.to_csv('output.csv', index=False)

output.head()