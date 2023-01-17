import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"

train_df = pd.read_csv(INPUT_PATH + "stage_1_train.csv")
label = train_df.Label.values

train_df = train_df.ID.str.rsplit("_", n=1, expand=True)

train_df.loc[:, "label"] = label
train_df = train_df.rename({0: "id", 1: "subtype"}, axis=1)

train_df.shape
train_pivot_df = pd.pivot_table(train_df, index="id", columns="subtype", values="label")
train_pivot_df.shape
train_pivot_df["normal"] = 1 - train_pivot_df["any"]
train_pivot_df.head()