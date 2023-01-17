import numpy as np

import pandas as pd

import seaborn as sns

from pathlib import Path

from PIL import Image



import plotly.graph_objects as go

import matplotlib.pyplot as plt
%ls ../input/bengaliai-cv19/
dataset_path = Path("../input/bengaliai-cv19")

num_dataset_count = 4

train_path = dataset_path / "train.csv"

test_path = dataset_path / "test.csv"

class_map_path = dataset_path / "class_map.csv "

sample_submit_path = dataset_path / "sample_submission.csv"



ext = ".parquet"

train_prefix = "train_image_data_"

test_prefix = "test_image_data_"

train_image_paths = []

test_image_paths = []

for idx in range(num_dataset_count):

    train_image_path = "{}{}{}".format(train_prefix, idx, ext)

    train_image_paths.append(train_image_path)

    test_image_path = "{}{}{}".format(test_prefix, idx, ext)

    test_image_paths.append(test_image_path)
train = pd.read_csv(train_path)

test = pd.read_csv(test_path)
train.head()
test.head()
train.shape, test.shape
first_target_col = "grapheme_root"

second_target_col = "vowel_diacritic"

third_target_col = "consonant_diacritic"
fig = go.Figure(data=[go.Histogram(x=train[first_target_col])])

fig.update_layout(title_text='{} value_counts'.format(first_target_col))

fig.show()
fig = go.Figure(data=[go.Histogram(x=train[second_target_col])])

fig.update_layout(title_text='{} value_counts'.format(second_target_col))

fig.show()
fig = go.Figure(data=[go.Histogram(x=train[third_target_col])])

fig.update_layout(title_text='{} value_counts'.format(third_target_col))

fig.show()
label_train = train.drop(columns=["image_id", "grapheme"])

label_train[label_train.duplicated()].shape
train_combs = train.groupby([first_target_col, second_target_col, third_target_col]).size().reset_index().rename(columns={0:'count'})
fig = go.Figure(data=[go.Bar(x=train_combs.index, y=train_combs["count"])])

fig.update_layout(title_text='unique combination value_counts')

fig.show()