import numpy as np

import pandas as pd

import glob

import matplotlib.pyplot as plt

import seaborn as sns
# Get a list of the data files

files = glob.glob("/kaggle/input/cicids2017/CICIDS-2017/CICIDS-2017/TrafficLabelling/*.csv")

files
# Load the data into a single dataframe

df = pd.DataFrame()



for file in files:

    df = pd.concat([df, pd.read_csv(file, encoding = "ISO-8859-1")])
df.info()
df.drop([" Source IP", " Destination IP", " Timestamp", "Flow ID"], inplace=True, axis=1)

df.head()
# Check for missing data

print(f"Missing values: {df.isnull().sum().sum()}")



# Check for infinite values, replace with NAN so it is easy to remove them

df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"Missing values: {df.isnull().sum().sum()}")
df.dropna(inplace=True)

df.shape
# Have a look at the attack categories

df[" Label"].unique()
# Rename some of the attacks, matplotlib has some trouble rendering the characters

df[" Label"] = df[" Label"].apply(lambda x: "Brute Force" if x=="Web Attack \x96 Brute Force" else x)

df[" Label"] = df[" Label"].apply(lambda x: "XSS" if x=="Web Attack \x96 XSS" else x)

df[" Label"] = df[" Label"].apply(lambda x: "SQL Injection" if x=="Web Attack \x96 Sql Injection" else x)



df["attack"] = df[" Label"].apply(lambda x: 0 if x=="BENIGN" else 1)

sns.countplot(x="attack", data=df)

plt.xticks([0,1], ["Normal", "Attack"])

plt.title("CICIDS2017 distribution prior to subsetting")

plt.xlabel("")

plt.show()
attacks = df[df["attack"]==1]

benign = df[df["attack"]==0]



print(f"Attack records: {len(attacks)}\nBenign records: {len(benign)}")
# Lets drop some of the benign data so we have a more balanced dataset

benign = benign.sample(frac=0.3).reset_index(drop=True)
print(f"Attack records: {len(attacks)}\nBenign records: {len(benign)}")
# Join the attacks and benign data back together

df = pd.concat([attacks, benign])

sns.countplot(x="attack", data=df)

plt.xticks([0,1], ["Normal", "Attack"])

plt.title("CICIDS2017 distribution after subsetting")

plt.xlabel("")

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df,

                                                    df["attack"], 

                                                    test_size=0.50, 

                                                    random_state=0,

                                                    stratify=df[" Label"],

                                                    shuffle=True)
# Visualise the distribution of attacks and normal traffic



f, axes = plt.subplots(2, 2, figsize=(12, 10))



# Create the plots

sns.countplot(x="attack", data=X_train, ax=axes[0,0])

sns.countplot(x="attack", data=X_test, ax=axes[0,1])

sns.countplot(x=" Label", data=X_train, ax=axes[1,0], order = X_train[' Label'].value_counts().index)

sns.countplot(x=" Label", data=X_test, ax=axes[1,1], order = X_test[' Label'].value_counts().index)



# Set the plot titles

axes[0,0].set_title("Training data distribution")

axes[1,0].set_title("Training data distribution")

axes[0,1].set_title("Testing data distribution")

axes[1,1].set_title("Testing data distribution")



# Rotate xticks for readability

axes[1,0].tick_params('x', labelrotation=90)

axes[1,1].tick_params('x', labelrotation=90)



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



# Drop the training targets from the training data

X_train.drop([" Label", "attack"], inplace=True, axis=1)

X_test.drop([" Label", "attack"], inplace=True, axis=1)
train = X_train.copy()

test = X_test.copy()



train.loc[:,"label"] = y_train

test.loc[:,"label"] = y_test



train.to_csv("/kaggle/working/train.csv", header=True, index=False)

test.to_csv("/kaggle/working/test.csv", header=True, index=False)
%%time

from sklearn.preprocessing import MinMaxScaler



# Create the scaler on the training data

scaler = MinMaxScaler()

scaler.fit_transform(X_train)



# Scale the testing data

scaler.transform(X_test)



from sklearn.tree import DecisionTreeClassifier

dtc_clf = DecisionTreeClassifier(random_state=0)

dtc_clf.fit(X_train, y_train)

print(f"Accuracy: {dtc_clf.score(X_test, y_test)}")