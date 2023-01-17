import numpy as np

import pandas as pd



import os, re



import seaborn as sns

import matplotlib.pyplot as plt

import random

sns.set(style="white", color_codes=True)

random_num = 10

%matplotlib inline
def summarize_categorical_col(df, col):

    print("----")

    print("column:", col)

    counts = df[col].value_counts()

    print("unique values:", len(counts))

    print("most frequent: %s (%s)" % (counts.index.values[0], str(counts.values[0])))

    print("null values:", df[col].isnull().sum())

    return counts
data_dir = "../input"

train_df = pd.read_csv(

    os.path.join(data_dir, "train.tsv"),

    delimiter = "\t",

    index_col='train_id'

).assign(

    log10_price = lambda df: np.log10(df["price"]+1)

)

print("number of rows", train_df.shape[0])

print("number of cols", train_df.shape[1])
train_df.sample(10, random_state=random_num)
desc = train_df.describe()

desc
for col in desc.columns.values:

    plt.figure()

    sns.distplot(train_df[col], kde=(col == "log10_price"))
categorical_cols = [col for col in train_df.columns.values if col not in desc.columns.values]

for col in categorical_cols:

    summarize_categorical_col(train_df, col)
processed = train_df.assign(

    categories = lambda df: df["category_name"].apply(lambda x: x.split("/") if isinstance(x, str) else None)

).pipe(

    lambda df: pd.concat(

        [df, df["categories"].apply(pd.Series)],

        axis = 1

    ).rename(

        columns=lambda col: "category"+str(col) if re.match(r'[0-9]+', str(col)) is not None else col

    )

)



processed.sample(10, random_state=random_num)
[

    summarize_categorical_col(processed, col) 

    for col 

    in processed.columns.values 

    if re.match(r'category[0-9]+', col) is not None

]

pass