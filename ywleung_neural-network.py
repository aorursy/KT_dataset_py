import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# load datasets
path = "../input/"
train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")

train_data.head()
# join train_data and test_data for data preprocessing
train_len = len(train_data)
dataset = pd.concat(objs = [train_data, test_data], axis = 0).reset_index(drop = True)

# fill empty and NaN values with NaN
dataset = dataset.fillna(np.nan)

# check for null value
dataset.isnull().sum()
# check data types and if any missing data
train_data.info()
train_data.isnull().sum()
train_data.describe()
dataset["Age"].isnull().sum()
# Explore Age against Sex, Parch, SibSp, Pclass
g = sns.factorplot(data = dataset, x = "Sex", y = "Age", kind = "box")
g = sns.factorplot(data = dataset, x = "Parch", y = "Age", kind = "box")
g = sns.factorplot(data = dataset, x = "SibSp", y = "Age", kind = "box")
g = sns.factorplot(data = dataset, x = "Pclass", y = "Age", kind = "box")
# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(),
                cmap = "coolwarm", annot = True)
# find out indices of observations with NaN age values
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

# fill in missing values
age_med = dataset["Age"].median()
for i in index_NaN_age:
    age_pred = dataset["Age"][((dataset["Pclass"] == dataset.iloc[i]["Pclass"]) &
                              (dataset["Parch"] == dataset.iloc[i]["Parch"]) &
                              (dataset["SibSp"] == dataset.iloc[i]["SibSp"]))].median()
    if np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_med
    else:
        dataset["Age"].iloc[i] = age_pred
g = sns.factorplot(data = train_data, x = "Survived", y = "Age", kind = "box")
g = sns.factorplot(data = train_data, x = "Survived", y = "Age", kind = "violin")
# fill missing value to "S"
dataset.loc[dataset["Embarked"].isnull(), "Embarked"] = "S"

# convert Embarked into categorical values
dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, "S": 2})

dataset["Embarked"] = dataset["Embarked"].astype(int)
dataset["family_size"] = dataset["Parch"] + dataset["SibSp"] + 1
g = sns.factorplot(data = dataset, x = "family_size", y = "Survived")
g.set_ylabels("Survival Rate")
# drop Parch, SibSp
dataset.drop(labels = ["Parch", "SibSp"], axis = 1, inplace = True)
# take a look at Name
dataset["Name"].head()
# get all titles from names
dataset_title = dataset["Name"].str.split(", ", expand = True)[1]
dataset_title = dataset_title.str.split(". ", expand = True)[0]

dataset_title.value_counts()
# find out titles with frequency < 10
vals, counts = np.unique(dataset_title, return_counts = True)
rare_title = vals[counts < 10]

print (rare_title)
# create a new feature title
dataset["title"] = dataset_title

# convert all titles to "Mr", "Miss", "Mrs", "Master", "Rare"
dataset["title"] = dataset["title"].replace(rare_title, "Rare")

# convert title to categorical value
dataset["title"] = dataset["title"].map({"Mr": 0, "Miss": 1, "Mrs": 2,
                                         "Master": 3, "Rare": 4})
dataset["title"] = dataset["title"].astype(int)
g = sns.countplot(dataset["title"])
g.set_xticklabels(["Mr", "Miss", "Mrs", "Master", "Rare"])
g = sns.factorplot(data = dataset, x = "title", y = "Survived", kind = "bar")
g.set_xticklabels(["Mr", "Miss", "Mrs", "Master", "Rare"])
g.set_ylabels("Survival Rate")
# Drop Name column
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
# remove Cabin
dataset.drop(labels = ["Cabin"], axis = 1, inplace = True)
dataset["Ticket"].head()
# remove ticket
dataset.drop(labels = ["Ticket"], axis = 1, inplace = True)
passengerId = dataset["PassengerId"]
# drop PassengerId
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.head()
# separate train set and test set
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels = ["Survived"], axis = 1, inplace = True)
# separate features and labels of train set
Y_train = train["Survived"].astype(int)
X_train = train.drop(labels = ["Survived"], axis = 1)
# convert Y_train into one-hot vector
Y_oh = np.eye(2)[Y_train]

Y_oh
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Model
input_shape = X_train.shape[1]
classes = Y_oh.shape[1]

input_X = Input((input_shape, ))
X = BatchNormalization()(input_X)
X = Dense(16, activation = "relu")(X)
X = Dropout(0.5)(X)
X = Dense(16, activation = "relu")(X)
X = Dropout(0.5)(X)
X = Dense(16, activation = "relu")(X)
X = Dense(classes, activation = "sigmoid")(X)

model = Model(inputs = input_X, outputs = X)
model.summary()
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
train_history = model.fit(x = X_train, y = Y_oh, validation_split = 0.2, 
                          epochs = 30)
plt.subplot(1, 2, 1)
plt.plot(train_history.history["acc"])
plt.plot(train_history.history["val_acc"])
plt.title("Accuracy")
plt.xlabel("epoch")

plt.subplot(1, 2, 2)
plt.plot(train_history.history["loss"])
plt.plot(train_history.history["val_loss"])
plt.title("Loss")
plt.xlabel("epoch")
pred = model.predict(test)
pred = np.argmax(pred, axis = 1)

nn_result = pd.concat([passengerId[train_len:].reset_index(drop = True),
                     pd.Series(pred)], axis = 1)

nn_result.to_csv("nn_result.csv", index = False)
