# データを取り込んでみます
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_uma = pd.read_csv('../input/uma.csv')
# 表示してみます
df_uma.describe()
