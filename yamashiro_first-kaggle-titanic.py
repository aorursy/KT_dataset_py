import pandas as pd

import numpy as np

import csv as csv

from sklearn.ensemble import RandomForestClassifier # 今回はランダムフォレストを使う



#訓練データの読み込み

train_df = pd.read_csv("../input/train.csv", header=0)



# Sexをダミー変数に変換(female = 0, Male = 1)

train_df["Gender"] = train_df["Sex"].map( {"female": 0, "male": 1} ).astype(int)

train_df.head(3)