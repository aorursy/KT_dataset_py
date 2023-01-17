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
pic_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
labels = pic_data['label']
print(test_data)
print(pic_data.head())
features = pic_data.iloc[:,1:]
print(labels)
print(features)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(features,labels)
test_data['Label'] = model.predict(test_data)
test_data['ImageId']=range(1,len(test_data)+1)
test_data[['ImageId','Label']].to_csv('submission.csv',header=True,index=False)
