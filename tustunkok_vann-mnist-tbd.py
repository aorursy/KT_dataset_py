import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder



from keras.layers import Dense, Input

from keras.models import Model, Sequential



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df.head()
X = df.drop("label", axis=1)

y = df["label"]
onehot_enc = OneHotEncoder()

y = onehot_enc.fit_transform(y.values.reshape(-1,1))
inputs = Input(shape=(784,))

hidden = Dense(128, activation="sigmoid")(inputs)

hidden = Dense(64, activation="sigmoid")(hidden)

hidden = Dense(32, activation="sigmoid")(hidden)

outputs = Dense(10, activation="softmax")(hidden)



model_func = Model(inputs=inputs, outputs=outputs)

model_func.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
model_func.summary()
model_func.fit(X, y, epochs=20)
predictions = onehot_enc.inverse_transform(model_func.predict(df_test.values))
submissions = dict()

submissions["ImageId"] = list(range(1, len(df_test) + 1))

submissions["Label"] = predictions.reshape(-1).astype(int).tolist()

submissions_df = pd.DataFrame(submissions)
submissions_df.to_csv("submission.csv", index=False)