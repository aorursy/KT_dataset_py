# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "-ltrah", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
loc = pd.read_csv("../input/Location.csv")
cat = pd.read_csv("../input/Category.csv")
sample = pd.read_csv("../input/Random_submission.csv")
train = pd.read_csv("../input/ItemPairs_train.csv")
test = pd.read_csv("../input/ItemPairs_test.csv")
testInfo = pd.read_csv("../input/ItemInfo_test.csv")
loc.shape
loc.regionID.value_counts()
cat.parentCategoryID.value_counts()
train.head()
test.head()
print(testInfo.columns)
print(testInfo.head(1))