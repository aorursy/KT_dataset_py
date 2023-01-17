# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#define global variables

DATA_PATH = "../input/jigsaw-multilingual-toxic-comment-classification"

small_ds_path = "jigsaw-toxic-comment-train.csv"

large_ds_path = "jigsaw-unintended-bias-train.csv"

val_ds_path = "validation.csv"

test_ds_path = "test.csv"
#download the data

small_ds = pd.read_csv(os.path.join(DATA_PATH, small_ds_path), usecols=["id", "comment_text", "toxic"])

large_ds = pd.read_csv(os.path.join(DATA_PATH, large_ds_path), usecols=["id", "comment_text", "toxic"])

val_ds = pd.read_csv(os.path.join(DATA_PATH, val_ds_path))

test_ds = pd.read_csv(os.path.join(DATA_PATH, test_ds_path))
small_ds.head()
vals = small_ds.toxic.value_counts()

sns.barplot(vals.index, vals.values)

plt.title("Non-toxic vs toxic occurence in data")

plt.ylabel("Number exmaples")

plt.xlabel("Target value")
toxic_examples = small_ds[small_ds["toxic"] == 1].sample(5, random_state=42)["comment_text"]

for comment in toxic_examples.values:

    print("Next comment:")

    print(comment)
#generate wordcloud to get more intuition about toxicity

from wordcloud import WordCloud

toxic_comments = " ".join(small_ds[small_ds["toxic"]==1]["comment_text"].values)

wc = WordCloud().generate(toxic_comments)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")
non_toxic_examples = small_ds[small_ds["toxic"] == 0].sample(5, random_state=42)["comment_text"]

for comment in non_toxic_examples.values:

    print("Next comment")

    print(comment)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

small_ds["num_words"] = small_ds["comment_text"].str.split().apply(len)

temp_ds = small_ds[small_ds["num_words"] < 500]

sns.violinplot(x="toxic",y="num_words", data=temp_ds, ax=ax1)

ax1.set_title("Distributions of number of words/sentences in toxic/nontoxic comments")



small_ds["num_sents"] = small_ds["comment_text"].str.split(".").apply(len)

temp2_ds = small_ds[small_ds["num_sents"] < 100]

sns.violinplot(x="toxic",y="num_sents", data=temp2_ds, ax=ax2)

#ax2.set_title("Distribution of number of sentences in toxic/nontoxic comments")

print("Number of words descriptive stats")

print(small_ds["num_words"].describe())

print()

print("Number of sentences descriptive stats")

print(small_ds["num_sents"].describe())
large_ds.head()
t = large_ds.toxic.round(1)

t.value_counts()
sns.barplot(t.value_counts().index, t.value_counts().values)



plt.ylabel("Num samples")

plt.xlabel("Probability of being toxic")
large_ds["rounded_toxic"] = large_ds.toxic.round(1)

maybe_toxic = large_ds[(large_ds.rounded_toxic == 0.5) | (large_ds.rounded_toxic == 0.6)].comment_text

probably_toxic = large_ds[(large_ds.rounded_toxic == 0.7) | (large_ds.rounded_toxic == 0.8)].comment_text

surely_toxic = large_ds[(large_ds.rounded_toxic == 0.9) | (large_ds.rounded_toxic == 1.0)].comment_text
# may be toxic examples



for comm in maybe_toxic.sample(3):

    print(comm)

    print()
maybe_toxic_comments = " ".join(maybe_toxic.values)

wc = WordCloud().generate(maybe_toxic_comments)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")
#probably toxic examples

for comm in probably_toxic.sample(3):

    print(comm)

    print()
probably_toxic_comments = " ".join(probably_toxic.values)

wc = WordCloud().generate(probably_toxic_comments)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")
#surely toxic examples

for comm in surely_toxic.sample(3):

    print(comm)

    print()
surely_toxic_comments = " ".join(surely_toxic.values)

wc = WordCloud().generate(surely_toxic_comments)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")
val_ds.head()
len(val_ds)
vs = val_ds.lang.value_counts()

sns.barplot(vs.index, vs.values)

plt.xlabel("language")

plt.ylabel("Number of samples")
ts = val_ds.toxic.value_counts()

sns.barplot(ts.index, ts.values)

plt.xlabel("Non-toxic vs Toxic")

plt.ylabel("Number of samples")
test_ds.head()
len(test_ds)
vc = test_ds.lang.value_counts()

sns.barplot(vc.index, vc.values)

plt.xlabel("language")

plt.ylabel("Number of samples")