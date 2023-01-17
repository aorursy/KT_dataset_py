import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load up the data

train_data = pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/UNSW_NB15_train-set.csv")

test_data =  pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/UNSW_NB15_test-set.csv")

features = pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/NUSW-NB15_features.csv", encoding = "ISO-8859-1")
# Print the number of train / test samples

print(f"Train data length: {len(train_data)}")

print(f"Test data length: {len(test_data)}")



# Visualise the distribution of attacks and normal traffic



f, axes = plt.subplots(2, 2, figsize=(12, 10))



# Create the plots

sns.countplot(x="label", data=train_data, ax=axes[0,0])

sns.countplot(x="label", data=test_data, ax=axes[0,1])

sns.countplot(x="attack_cat", data=train_data, ax=axes[1,0], order = train_data['attack_cat'].value_counts().index)

sns.countplot(x="attack_cat", data=test_data, ax=axes[1,1], order = test_data['attack_cat'].value_counts().index)



# Set the plot titles

axes[0,0].set_title("Training data distribution")

axes[1,0].set_title("Training data distribution")

axes[0,1].set_title("Testing data distribution")

axes[1,1].set_title("Testing data distribution")



# Rotate xticks for readability

axes[1,0].tick_params('x', labelrotation=45)

axes[1,1].tick_params('x', labelrotation=45)



# Change the xtick labels for attack / normal

axes[0,0].set_xticklabels(["Normal", "Attack"])

axes[0,1].set_xticklabels(["Normal", "Attack"])



# Remove xlabels

axes[0,0].set_xlabel("")

axes[0,1].set_xlabel("")

axes[1,0].set_xlabel("")

axes[1,1].set_xlabel("")



# Add some space between the plots for y labels

plt.subplots_adjust(wspace=0.25)

features
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
df = pd.concat([train_data, test_data], ignore_index=True)



# Remove unwanted columns

df.drop(['id', 'attack_cat'], inplace=True, axis=1)



# Perform one-hot encoding on categorical columns and join back to main train_data

one_hot = pd.get_dummies(df[["proto", "state", "service"]])

df = df.join(one_hot)



# Remove the original categorical columns

df.drop(["proto", "state", "service"], inplace=True, axis=1)



# Re split the data back into train / test

train_data = df.iloc[0:175341, 0:]

test_data = df.iloc[175341:, 0:]



# Create y_train and then drop the label from the training data

y_train = np.array(train_data["label"])

train_data.drop(['label'], inplace=True, axis=1)



y_test = np.array(test_data["label"])

test_data.drop(['label'], inplace=True, axis=1)



# Use min-max scaler to scale the features to 0-1 range

# Only fit the scaler on the train data!!

scaler = MinMaxScaler()

X_train = scaler.fit_transform(train_data)



# Scale the testing data

X_test = scaler.transform(test_data)



# Ensure our dataset splits are still correct

print(f"Train data shape: {X_train.shape} Train label shape: {y_train.shape}")

print(f"Test data shape: {X_test.shape} Test label shape: {y_test.shape}")
from sklearn.tree import DecisionTreeClassifier

dtc_clf = DecisionTreeClassifier(random_state=0)

dtc_clf.fit(X_train, y_train)

print(f"Accuracy: {dtc_clf.score(X_test, y_test)}")