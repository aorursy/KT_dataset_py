%matplotlib inline
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# get titanic & test csv files as a DataFrame
gifts_df = pd.read_csv("../input/gifts.csv", dtype={"Age": np.float64}, )

# preview the data
gifts_df.head()
gifts_df.info()
giftsort = gifts_df.sort_values(by="Weight")
print(giftsort.tail())
giftsort['Weight'].plot(kind='hist')
giftsort['Weight'].median()
giftsort['Weight'].mean()
