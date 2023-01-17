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
df = pd.read_csv("../input/train.csv")
print(df.head())
#check out each column and find missing values 
print( df.describe() )
#270 missing value in age
#SibSp ?? 
#parch ??
# fair zeros
print("num of null is ", len(df.Embarked[ df.Embarked.isnull() ]));
print( df.isnull().sum(0) );
from sklearn.ensemble import RandomForestClassifier

print("training")
df["gender"] = df["Sex"].map( { "male" : 0, "female" : 1 } ).astype(int) 
df.dropna()
print(df.describe())

forest = RandomForestClassifier( n_estimators=100)

forest = forest.fit(df[["Pclass", "Fare"]], df["Survived"] )

df_test = pd.read_csv("../input/test.csv");
print(df_test.describe())
df_test.dropna()
df_test["gender"] = df_test["Sex"].map( { "male" : 0, "female" : 1 } ).astype(int) 
pred = forest.predict( df_test[["Pclass", "Fare"]] )
#print(pred)

