import numpy as np
import os
from pathlib import Path
import pandas as pd
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn import svm
p = Path("../input/pokemon/Train/Images")
dirs = p.glob("*")
x = pd.read_csv("../input/pokemon/Train/train.csv").values
x = x[:,1]
y = []
for label in x:
    if (label == 'Pikachu'):
        y.append(0)
    elif (label == 'Bulbasaur'):
        y.append(1)
    elif (label == 'Charmander'):
        y.append(2)
print(y)
img_data = []
for image_name in dirs:
    img = image.load_img(image_name, target_size = (100, 100))
    img_array = image.img_to_array(img)/255
    img_data.append(img_array)
# print(img_data)
#print(img_data[0].shape)
img_data = np.array(img_data)
label = np.array(y)
M = img_data.shape[0]
img_data = img_data.reshape(M, -1)
print(img_data.shape)
p = Path("../input/pokemon/Test/Images")
dirs = p.glob("*")
test_data = []
t = []
for image_name in dirs:
    im = str(image_name).split("\\")[-1]
    t.append(im)
    img = image.load_img(image_name, target_size = (100, 100))
    img_array = image.img_to_array(img)/255
    test_data.append(img_array)
test_data = np.array(test_data)
n = test_data.shape[0]
test_data = test_data.reshape(n, -1)
print(n)
print(test_data.shape)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(img_data, label)

y = logisticRegr.predict(test_data)
print(y)
test = pd.read_csv("../input/pokemon/Test/test.csv").values

print(test)
yt = []
for tes in y:
    if tes == 0:
        yt.append('Pikachu')
    elif tes == 1:
        yt.append('Bulbasaur')
    else:
        yt.append('Charmander')
print(yt)
yt = np.array(yt)

out = list(zip(test, yt))
# output = []
# for i in range(123):
#     for j in range(123):
#         if test[i] == t[j]:
#             output.append(out[j])
#             break
output = np.array(out)
df = pd.DataFrame(output, columns = ["ImageId", "NameOfPokemon"])
df.to_csv('output.csv', index = False)
print(df)
out = np.array(out)
df = pd.DataFrame(out, columns = ["ImageId", "NameOfPokemon"])
df.to_csv('output.csv', index = False)
print(df)