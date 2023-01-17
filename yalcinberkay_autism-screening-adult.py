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
df = pd.read_csv("../input/Autism-Adult-Data.csv")

df.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use("bmh")
#Collection information about dataset

df.info()
print(df["Class/ASD"].describe())

plt.figure(figsize=(9,8))

sns.countplot(x="Class/ASD",data=df)
#Numeric ve Categorical featureları tek tek inceleyeceğiz.

#Aşağıdaki göreceğimiz üzere int64 ve Object tipinde iki tür featureımız var

list(set(df.dtypes.tolist()))
df_numeric = df.select_dtypes(include=["int64"])

df_numeric.head()
df_numeric.hist(figsize=(16,20),bins=50,xlabelsize=8,ylabelsize=8)
df_numeric_correlation = df_numeric.corr()["result"][:-1]

golden_features_list = df_numeric_correlation[abs(df_numeric_correlation) > 0.5].sort_values(ascending=False)

print("There is {} strongly correlated values with class :\n{}".format(len(golden_features_list),golden_features_list))
df_objects = df.select_dtypes(include=["O"])

df_objects.head()
plt.figure(figsize = (20,20))

df_for_count = df.copy()

# delete = df_for_count[df_for_count["age"]=="?"]

df_for_count.drop(df_for_count.loc[df_for_count["age"]=="?","age"].index,inplace=True)

df_for_count["age"] = ["under 35" if age<=35 else "higher 35" if age>=35 else "None" for age in df_for_count["age"].astype("int64")]



# ["under 35" if type(age) == "int64" and age <= 35 else 'higher 35' if v == 2 else 'None' for v in l]

    

ax = sns.countplot(x="Class/ASD",hue="age",data=df_for_count)

plt.setp(ax.artists,alpha=.5,linewidth=2,edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (20,20))

df_for_count = df.copy()

# delete = df_for_count[df_for_count["age"]=="?"]

df_for_count.drop(df_for_count.loc[df_for_count["gender"]=="?","gender"].index,inplace=True)



# ["under 35" if type(age) == "int64" and age <= 35 else 'higher 35' if v == 2 else 'None' for v in l]

    

ax = sns.countplot(x="Class/ASD",hue="gender",data=df_for_count)

plt.setp(ax.artists,alpha=.5,linewidth=2,edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (20,20))

df_for_count = df.copy()

df_for_count.drop(df_for_count.loc[df_for_count["ethnicity"]=="?","ethnicity"].index,inplace=True)    

ax = sns.countplot(x="Class/ASD",hue="ethnicity",data=df_for_count)

plt.setp(ax.artists,alpha=.5,linewidth=2,edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (35,35))

df_for_count = df.copy()

df_for_count.drop(df_for_count.loc[df_for_count["contry_of_res"]=="?","contry_of_res"].index,inplace=True)    

ax = sns.countplot(x="Class/ASD",hue="contry_of_res",data=df_for_count,order = df_for_count['Class/ASD'].value_counts().index)

plt.setp(ax.artists,alpha=.5,linewidth=2,edgecolor="k")

plt.xticks(rotation=45)

df.head()
df = df.replace('?', np.nan)

df.head()
df = df.fillna(df.mode().iloc[0])

df.head()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['ethnicity'] = le.fit_transform(df['ethnicity'])

df['jundice'] = le.fit_transform(df['jundice'])

df['austim'] = le.fit_transform(df['austim'])

df['contry_of_res'] = le.fit_transform(df['contry_of_res'])

df['age_desc'] = le.fit_transform(df['age_desc'])

df['relation'] = le.fit_transform(df['relation'])

df['Class/ASD'] = le.fit_transform(df['Class/ASD'])

df['used_app_before'] = le.fit_transform(df['used_app_before'])

df['gender'] = le.fit_transform(df['gender'])

df['age'] = le.fit_transform(df['age'])

df.head()
df.info()
X = df.drop("Class/ASD",axis=1)

Y = df["Class/ASD"]

X,Y
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(categories="auto")

X = ohe.fit_transform(X).toarray()

X

print("Our Sample Len : {} and Features of NN {}".format(X.shape[0],X.shape[1]))

print("Our Outputs Len : {}".format(Y.shape[0]))
import tensorflow as tf
def build_model():

    inputs = tf.keras.Input(shape=(169,))

    x = tf.keras.layers.Dense(128,activation=tf.nn.relu)(inputs)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(256,activation=tf.nn.relu)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(512,activation=tf.nn.relu)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    x = tf.keras.layers.Dense(1024,activation=tf.nn.relu)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)

    logits = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)(x)



    model = tf.keras.Model(inputs=inputs,outputs=logits)

    optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    loss = tf.keras.losses.binary_crossentropy

    model.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])

    return model

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state = 42)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True)

for index, (train_indices, val_indices) in enumerate(skf.split(X_train, Y_train)):

    print("Training on fold" + str(index+1) + "/10...")

    # Generate batches from indices

    xtrain, xval = X_train[train_indices], X_train[val_indices]

    ytrain, yval = Y_train[train_indices], Y_train[val_indices]

    # Clear model, and create it

    model = None

    model = build_model()

    

    # Debug message I guess

    print( "Training new iteration on " + str(xtrain.shape[0]) + " training samples, " + str(xval.shape[0]) + " validation samples, this may be a while...")

    

    history = model.fit(xtrain,ytrain,epochs=20,batch_size=32,validation_data=(xval,yval))

    accuracy_history = history.history['acc']

    val_accuracy_history = history.history['val_acc']

    print("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]))
scores = model.evaluate(x=X_test,y=Y_test,batch_size=32)

print(scores)

y_head = model.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test,y_head.round())
print(cm)
TP , FP , FN , TN = cm[0][0] ,cm[0][1] , cm[1][0] , cm[1][1]
print("Class 0 Prediction Accuracy {:.1f}".format(TP / (TP+FP) * 100))

print("Class 1 Prediction Accuracy {:.1f}".format(TN / (FN+TN) * 100))