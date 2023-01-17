import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_plus = pd.read_csv("/kaggle/input/nslkdd/KDDTrain+.txt", header=None)

test_plus = pd.read_csv("/kaggle/input/nslkdd/KDDTest+.txt", header=None)

test_minus_twentyone = pd.read_csv("/kaggle/input/nslkdd/KDDTest-21.txt", header=None)
# Check the shape of the data

print(f"Shape of train plus: {train_plus.shape}")

print(f"Shape of test plus: {test_plus.shape}")

print(f"Shape of test-21: {test_minus_twentyone.shape}")



# Check for missing values

print(f"\nMissing values in train plus: {train_plus.isnull().sum().sum()}")

print(f"Missing values in test plus: {test_plus.isnull().sum().sum()}")

print(f"Missing values in train plus: {test_minus_twentyone.isnull().sum().sum()}")
colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 

            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 

            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',

            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 

            'srv_rerror_rate',  'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 

            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',

            'dst_host_srv_rerror_rate', 'class']



print(len(colnames))
# We are provided 42 column names. 

# We have 43 columns in the .txt files. 

# This extra column is dropped because no information is provided



train_plus.drop(42, axis=1, inplace=True)

test_plus.drop(42, axis=1, inplace=True)

test_minus_twentyone.drop(42, axis=1, inplace=True)



train_plus.columns = colnames

test_plus.columns = colnames

test_minus_twentyone.columns = colnames



train_plus.info()
# Concat the df's together

df = pd.concat([train_plus, test_plus, test_minus_twentyone], ignore_index=True)



# Convert the attack label to a binary classification problem 0=normal 1=attack

df["attack"] = df["class"].apply(lambda x: 0 if x=="normal" else 1)



# Get the one-hot encoding

one_hot = pd.get_dummies(df[["protocol_type", "service", "flag"]])

df = df.join(one_hot)

df.drop(["protocol_type", "service", "flag"], inplace=True, axis=1)



# resplit the data

train_plus = df.iloc[0:125973, :]

test_plus = df.iloc[125973:148517, :]

test_minus_twentyone = df.iloc[148517:,:]



# Check the shape of the data

print(f"Shape of train plus: {train_plus.shape}")

print(f"Shape of test plus: {test_plus.shape}")

print(f"Shape of test-21: {test_minus_twentyone.shape}")
# Visualise the distribution of attacks and normal traffic



f, axes = plt.subplots(2, 3, figsize=(18, 10))



# Create the plots

sns.countplot(x="attack", data=train_plus, ax=axes[0,0])

sns.countplot(x="attack", data=test_plus, ax=axes[0,1])

sns.countplot(x="attack", data=test_minus_twentyone, ax=axes[0,2])

sns.countplot(x="class", data=train_plus, ax=axes[1,0], order = train_plus['class'].value_counts().index, palette="tab10")

sns.countplot(x="class", data=test_plus, ax=axes[1,1], order = test_plus['class'].value_counts().index, palette="tab10")

sns.countplot(x="class", data=test_minus_twentyone, ax=axes[1,2], order = test_minus_twentyone['class'].value_counts().index, palette="tab10")



# Set the plot titles

axes[0,0].set_title("Train+ data distribution")

axes[1,0].set_title("Train+ data distribution")

axes[0,1].set_title("Test+ data distribution")

axes[0,2].set_title("Test-21 data distribution")

axes[1,1].set_title("Test+ data distribution")

axes[1,2].set_title("Test-21 data distribution")



# Rotate xticks for readability

axes[1,0].tick_params('x', labelrotation=90)

axes[1,1].tick_params('x', labelrotation=90)

axes[1,2].tick_params('x', labelrotation=90)



# Change the xtick labels for attack / normal

axes[0,0].set_xticklabels(["Normal", "Attack"])

axes[0,1].set_xticklabels(["Normal", "Attack"])

axes[0,2].set_xticklabels(["Normal", "Attack"])



# Remove xlabels

axes[0,0].set_xlabel("")

axes[0,1].set_xlabel("")

axes[0,2].set_xlabel("")

axes[1,0].set_xlabel("")

axes[1,1].set_xlabel("")



# Add some space between the plots for y labels

plt.subplots_adjust(wspace=0.25)
from sklearn.preprocessing import StandardScaler



# Create the training target for each dataset

y_train = np.array(train_plus["attack"])

y_test = np.array(test_plus["attack"])

y_test_minus_twentyone = np.array(test_minus_twentyone["attack"])



# Create the scaler and fit only on the training data, drop our labels for X_train.

# train_plus.drop(["attack", "class"], axis=1, inplace=True)

# test_plus.drop(["attack", "class"], axis=1, inplace=True)

# test_minus_twentyone.drop(["attack", "class"], axis=1, inplace=True)



scaler = StandardScaler()

X_train = scaler.fit_transform(train_plus.drop(["attack", "class"], axis=1))

X_test = scaler.transform(test_plus.drop(["attack", "class"], axis=1))

X_test_minus_twentyone = scaler.transform(test_minus_twentyone.drop(["attack", "class"], axis=1))



print(f"Shape for X_train: {X_train.shape} y_train: {y_train.shape}")

print(f"Shape for X_test: {X_test.shape} y_test: {y_test.shape}")

print(f"Shape for X_test_minus_twentyone: {X_test_minus_twentyone.shape} y_train: {y_test_minus_twentyone.shape}")
%%time



from sklearn.tree import DecisionTreeClassifier

dtc_clf = DecisionTreeClassifier(random_state=0)

dtc_clf.fit(X_train, y_train)

print(f"Accuracy on test+: {dtc_clf.score(X_test, y_test)}")

print(f"Accuracy on test-21: {dtc_clf.score(X_test_minus_twentyone, y_test_minus_twentyone)}")