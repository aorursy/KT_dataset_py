# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/inc_occ_gender.csv") # load the target data
# df = df.dropna() # 失敗　NaNじゃない、Naだ
df = df[df.M_weekly != "Na"] # M_weeklyでNaのrowを除く
df = df[df.F_weekly != "Na"] # F_weeklyでNaのrowを除く
df["M_weekly"]=df["M_weekly"].astype(pd.np.float64)
df["F_weekly"]=df["F_weekly"].astype(pd.np.float64)
df['W_diff'] = (df["M_weekly"] - df["F_weekly"]) 
df = df.sort_values(by='W_diff', ascending=False)
df
import seaborn as sns

df = df[df.W_diff > 450] 

sns.barplot(x="W_diff", y="Occupation" , data=df)