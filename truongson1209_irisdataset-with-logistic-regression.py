# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import pandas as pd     # Khai báo các thư viện thường dùng
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split   #Chia tập dữ liệu thành tập train và test
from sklearn.linear_model import LogisticRegression    # chọn bài toán logistic regression 
from sklearn.metrics import accuracy_score             #Lấy công cụ đánh giá
import matplotlib.pyplot as plt
#get data from .csv file
data=pd.read_csv("/kaggle/input/iris/Iris.csv")
encode=LabelEncoder()
data.Species=encode.fit_transform(data.Species)
train,test=train_test_split(data,test_size=0.3,random_state=0)
train_x=train.drop(columns=["Species"],axis=1)
train_y=train["Species"]
test_x=test.drop(columns=["Species"],axis=1)
test_y=test["Species"]
#Tạo mô hình
model=LogisticRegression()
model.fit(train_x,train_y,sample_weight=None)
#Evaluation
predict=model.predict(test_x)

print("\nAccuracy Score",accuracy_score(test_y,predict))