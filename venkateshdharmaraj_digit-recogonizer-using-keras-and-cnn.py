# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
x_train = train.drop('label',axis=1)
y_train = train['label']
x_train=x_train.values.reshape(-1,28,28,1)
y_train.values.reshape(-1,1)

x_test = test
x_test=x_test.values.reshape(-1,28,28,1)
x_train=x_train/255#if you run the above cell you get to know that all the values lies between 0-255 for each traing eg. So,in order to make the values to range between 0-1 we divide it by 255
x_test=x_test/255
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64,activation="relu",input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation="softmax"),
    ]
)
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, batch_size =64, epochs = 20, shuffle = True)
model.save("digit_recogonizer.model")
newmodel=tf.keras.models.load_model("digit_recogonizer.model")
predictions=newmodel.predict([x_test])
print(predictions)
np.argmax(predictions[0])#prediction of the first test sample
#checking whether our test sample is sample as the above prediction
image=x_test[0]
image=np.reshape(image,(28,28))#reshaping our one dimensional vector back to original dimensional
plt.imshow(image,cmap="hot")
predictclass=model.predict_classes(x_test)
predictclass
final= pd.DataFrame([test.index+1,predictclass],["ImageId","Label"]).T
final.to_csv('digit_recogonizer.csv',index=False)