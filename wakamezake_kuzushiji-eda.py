import re

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

input_path = Path("../input")

%matplotlib inline
# reference

# https://www.kaggle.com/anokas/kuzushiji-visualisation

df_train = pd.read_csv( input_path / 'train.csv')

unicode_trans = pd.read_csv( input_path / 'unicode_translation.csv')

train_image_path = input_path / "train_images"

test_image_path = input_path / "test_images"

unicode_map = {codepoint: char for codepoint, char in unicode_trans.values}
# nan count per column

df_train.isnull().sum()
# Show NaN column

nan_df = df_train[df_train.isnull()["labels"] == True]

nan_df.head()
# get unique labels

length = 5

labels = []

for label in df_train["labels"]:

    # skip nan

    if type(label) == str:

        split_label = label.split()[::length]

        labels += split_label
print("Number of unique_label: {}".format(len(set(labels))))

print("Number of unique_label(unicode_translation.csv): {}".format(unicode_trans.shape[0]))

print("diff: {}".format(abs(len(set(labels)) - unicode_trans.shape[0])))
unicode_trans_only_labels = set(unicode_trans["Unicode"]) - set(labels)

print("Number of unicode_trans_only_label: {}".format(len(unicode_trans_only_labels)))
unicode_trans_only_df = pd.DataFrame({"Unicode": list(unicode_trans_only_labels)})

unicode_trans_only_df["string"] = unicode_trans_only_df["Unicode"].map(unicode_map)

unicode_trans_only_df.head()
df_labels = pd.DataFrame({"label": labels})

# df_labels["str"] = df_labels["label"].map(unicode_map)

label_count  = df_labels["label"].value_counts()
# Reference

# https://www.kaggle.com/tejainece/seaborn-barplot-and-pandas-value-counts

# label_count  = df_labels["label"].value_counts()

# plt.figure(figsize=(16,10))

# sns.barplot(label_count.index, label_count.values, alpha=0.8)

# plt.title('all unicode counts')

# plt.ylabel('Number of count', fontsize=12)

# plt.xlabel('unicode', fontsize=12)

# plt.show()
# Reference

# https://www.kaggle.com/tejainece/seaborn-barplot-and-pandas-value-counts

plt.figure(figsize=(10,5))

sns.barplot(label_count[:10,].index, label_count[:10,].values, alpha=0.8)

plt.title('top 10 unicode counts')

plt.ylabel('Number of count', fontsize=12)

plt.xlabel('unicode', fontsize=12)

plt.show()
# convert unicode to str

d = {"unicode": label_count[:10].index.values,

     "str": [unicode_map[l] for l in label_count[:10].index.values]}

pd.DataFrame(d)
plt.figure(figsize=(10,5))

sns.barplot(label_count[-11:-1,].index, label_count[-11:-1,].values, alpha=0.8)

plt.title('bottom 10 unicode counts')

plt.ylabel('Number of count', fontsize=12)

plt.xlabel('unicode', fontsize=12)

plt.show()
# convert unicode to str

d = {"unicode": label_count[-11:-1,].index.values,

     "str": [unicode_map[l] for l in label_count[-11:-1,].index.values]}

pd.DataFrame(d)
# Reference

# https://ja.wikipedia.org/wiki/%E5%B9%B3%E4%BB%AE%E5%90%8D_(Unicode%E3%81%AE%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF)

# > U+3040..U+309F

hirakana = re.compile(r"U\+30[4-9][0-9a-fA-F]")
sum(label_count)
# The appearance rate of Hirakana accounts for about 60% of the whole.

hirakana_counts = df_labels["label"].str.match(hirakana).value_counts()

hirakana_counts / sum(hirakana_counts) * 100
df_labels["is_hirakana"] = df_labels["label"].str.match(hirakana)

label_count = df_labels[df_labels["is_hirakana"] == True]["label"].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(label_count[-11:-1,].index, label_count[-11:-1,].values, alpha=0.8)

plt.title('hirakana bottom 10 unicode counts')

plt.ylabel('Number of count', fontsize=12)

plt.xlabel('unicode', fontsize=12)

plt.show()
# convert unicode to str

d = {"unicode": label_count[-11:-1,].index.values,

     "str": [unicode_map[l] for l in label_count[-11:-1,].index.values]}

pd.DataFrame(d)
label_count = df_labels[df_labels["is_hirakana"] == False]["label"].value_counts()
plt.figure(figsize=(10,5))

sns.barplot(label_count[:10,].index, label_count[:10,].values, alpha=0.8)

plt.title('top 10 unicode counts')

plt.ylabel('Number of count', fontsize=12)

plt.xlabel('unicode', fontsize=12)

plt.show()
# convert unicode to str

d = {"unicode": label_count[:10,].index.values,

     "str": [unicode_map[l] for l in label_count[:10,].index.values]}

pd.DataFrame(d)
plt.figure(figsize=(10,5))

sns.barplot(label_count[-11:-1,].index, label_count[-11:-1,].values, alpha=0.8)

plt.title('hirakana bottom 10 unicode counts')

plt.ylabel('Number of count', fontsize=12)

plt.xlabel('unicode', fontsize=12)

plt.show()
# convert unicode to str

d = {"unicode": label_count[-11:-1,].index.values,

     "str": [unicode_map[l] for l in label_count[-11:-1,].index.values]}

pd.DataFrame(d)