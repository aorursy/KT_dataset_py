import pandas as pd

import pandas_profiling
train = pd.read_csv("/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv")
train.shape
pandas_profiling.ProfileReport(train)