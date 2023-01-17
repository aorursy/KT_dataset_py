import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline
raw_data = pd.read_csv("../input/train.csv")

raw_data.head(5)
data = raw_data.drop("label",axis=1)

label = raw_data["label"]
image_data = data.iloc[5]

plt.imshow(image_data.values.reshape(28,28), cmap="Greens")
temp = data.iloc[5]

temp = temp/255

plt.imshow(temp.values.reshape(28,28), cmap="Greens")
temp1 = data.iloc[5]

temp1[temp1>1]=1

plt.imshow(temp1.values.reshape(28,28), cmap="Greens")
from sklearn.svm import SVC
model = SVC(gamma='scale')
data[data>0]=1
model.fit(data, label)
test_data = pd.read_csv("../input/test.csv")
test_data[test_data>0]=1
prediction = model.predict(test_data)
results = pd.Series(prediction,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("results.csv",index=False)