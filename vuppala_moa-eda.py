# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_features.shape
train_features.head()
train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_targets_scored.head()
train_targets_scored.shape
train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")
print(train_targets_nonscored.shape),
train_targets_nonscored.head()
print(train_targets_scored.shape),
train_targets_scored.head()
test_features = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
print(test_features.shape)
test_features.head()
train_features.cp_type.value_counts(dropna = False)
test_features.cp_type.value_counts(dropna = False)
test_features.shape
