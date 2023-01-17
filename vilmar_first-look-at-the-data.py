import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv('../input/training_variants')

print('Size of training set: {} rows and {} columns'.format(*df_train.shape))

df_train.head()
df_train["Class"].unique()
plt.figure(figsize=(8,5))

sns.countplot(x="Class", data=df_train)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Class Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Class Distribution", fontsize=15)

plt.show()
print("{} unique Gene".format(len(df_train["Gene"].unique())))

print("{} unique Variation".format(len(df_train["Variation"].unique())))
value_counts = df_train["Variation"].value_counts()

value_counts.head(20)
df_test = pd.read_csv('../input/test_variants')

print('Size of test set: {} rows and {} columns'.format(*df_test.shape))

df_test.head()
print("{} unique Gene in test set".format(len(df_test["Gene"].unique())))

print("{} unique Variation in test set".format(len(df_test["Variation"].unique())))
print("Number of common genes: {}".format(len(set(df_train["Gene"]).intersection(df_test["Gene"]))))

print("Number of common variations: {}".format(len(set(df_train["Variation"]).intersection(df_test["Variation"]))))