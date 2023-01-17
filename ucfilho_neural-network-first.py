import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
prop_df = pd.read_csv("../input/properties_2016.csv")

train_df = pd.read_csv("../input/train_2016.csv")

data= pd.merge(left=train_df,right=prop_df, left_on='parcelid', right_on='parcelid') # merge target and train
# split the dataset (train and test)

train=data.sample(frac=0.4,random_state=200)

test=data.drop(train.index)
# geting the name of variables

strings = list(data)
# dataset: features to be excluded 

invalid = ['parecelid', 'logerror', 'transactiondate','propertycountylandusecode','propertyzoningdesc','taxdelinquencyflag',

               'hashottuborspa','fireplaceflag','latitude','longitude']
# exlucluding features

Nomes = list(filter(lambda x: not any(s in x.lower() for s in invalid),strings))
#s plit the dataset in train and target and replace NaNs by zero

train_x=train[Nomes]

train_x[np.isnan(train_x)] = 0

train_y=train[['logerror']]
# normatization of dataset

train_y=(train_y-train_y.min())/(train_y.max()-train_y.min())

train_x=(train_x-train_x.min())/(train_x.max()-train_x.min())
#create neural net regressor

# activation : {'identity', 'logistic', 'tanh', 'relu'},

reg = MLPRegressor(hidden_layer_sizes=(100,),activation='logistic',solver="lbfgs",max_iter=10000)

reg.fit(train_x,train_y)

 

#test prediction

test_x=train_x

predict=reg.predict(test_x)
%matplotlib inline

plt.figure(figsize=(8,6))

plt.scatter(train_y,predict)

plt.xlabel("y_train")

plt.ylabel("y_calc")

plt.xlim((0,1))

plt.ylim((0,1))

plt.title("my first ANN approx")



plt.show()
