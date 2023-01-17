# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
glove_submission = pd.read_csv("../input/bi-gru-cnn-poolings-gpu-kernel-version/submission.csv")
fasttext_submission = pd.read_csv("../input/bi-gru-lstm-cnn-poolings-fasttext/submission.csv")
glove_submission.head()
fasttext_submission.head()
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
blend = fasttext_submission.copy()
blend[categories] = (0.5*fasttext_submission[categories].values +
                     0.5*glove_submission[categories].values)
blend.head()
blend.to_csv("blend.csv", index=False)
