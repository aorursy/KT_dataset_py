# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris_data = pd.read_csv("../input/Iris.csv")
iris_data.head()
iris_data.columns
iris_data['Species'].nunique()
iris_data.count()
iris_data['Species'].value_counts()
iris_data.describe()

iris_data['Species'].value_counts().plot.bar()
pd.plotting.parallel_coordinates(iris_data, "Species")
pd.plotting.parallel_coordinates(iris_data.drop('Id', axis=1), "Species")#不要id
import seaborn as sns
#在这个散点图矩阵中，我们可以确认  Setosa物种是线性可分的，如果你想要，你可以在这个物种和另外两个物种之间绘制一条分界线。但是，Versicolor和Virginica  类不是线性可分的。
sns.pairplot(iris_data.drop('Id', axis=1), hue='Species')
#用于可视化多元数据集群的Andrew Curves图。我们可以使用
pd.plotting.andrews_curves(iris_data.drop('Id', axis=1), 'Species')
X = iris_data.iloc[:, 1:5].values
#我们可以使用pandas.Series.cat.codes   来获取类标签的数值。
y = iris_data['Species'].astype('category').cat.codes
# print(y)
#我们已经完成了将字母数字类转换为数字类的第一步。现在要转换为One-hot编码，
Y = keras.utils.to_categorical(y, num_classes=None)
# print(Y)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


model = keras.Sequential()
model.add(keras.layers.Dense(4, input_shape=(4,), activation='tanh'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(keras.optimizers.Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

# model.fit(X_train, y_train, epochs=100)# 输出过程信息
model.fit(X_train, y_train, epochs=100, verbose=0)
accuracy = model.evaluate(X_test, y_test)[1]
print('Accuracy: {}'.format(accuracy))
results_control_accuracy = []
for i in range(0, 30):
    model = keras.Sequential()
    model.add(keras.layers.Dense(4, input_shape=(4,), activation='tanh'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(keras.optimizers.Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100, verbose=0)
    accuracy = model.evaluate(X_test, y_test)[1]
    results_control_accuracy.append(accuracy)
print(results_control_accuracy)
results_experimental_accuracy = []
for i in range(0, 30):
    model = keras.Sequential()
    #the number of neurons on the hidden layer. 增加到5个看看效果 
    model.add(keras.layers.Dense(5, input_shape=(4,), activation='tanh'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(keras.optimizers.Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=100, verbose=0)
    accuracy = model.evaluate(X_test, y_test)[1]
    results_experimental_accuracy.append(accuracy)
print(results_experimental_accuracy)
pd.DataFrame(results_control_accuracy).to_csv('results_control_accuracy.csv')
pd.DataFrame(results_experimental_accuracy).to_csv('results_experimental_accuracy.csv')
print(os.listdir('../input/iris_classifier_by_neural_network/'))
results_control_accuracy = pd.read_csv('../input/iris_classifier_by_neural_network/results_control_accuracy.csv')
results_experimental_accuracy = pd.read_csv('../input/iris_classifier_by_neural_network/results_experimental_accuracy.csv')

results_control_accuracy = pd.DataFrame([0.9333333359824286, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9555555568801032, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.6000000052981906, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9111111124356588, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9555555568801032, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9111111124356588])

results_experimental_accuracy = pd.DataFrame([0.9111111124356588, 0.9555555568801032, 0.9555555568801032, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9555555568801032, 0.933333334657881, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9555555568801032, 0.9777777791023254, 0.933333334657881, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9777777791023254, 0.9333333359824286, 0.9777777791023254, 0.9777777791023254, 0.9333333359824286, 0.9777777791023254, 0.9555555568801032, 0.9777777791023254, 0.9777777791023254])
print("Mean control accuracy = {}".format(results_control_accuracy.mean()))
print("Mean experimental accuracy = {}".format(results_experimental_accuracy.mean()))
results_accuracy = pd.concat([results_control_accuracy, results_experimental_accuracy], axis=1)
results_accuracy.columns = ['Control', 'Experimental']
# print(results_accuracy)
results_accuracy.boxplot()

results_accuracy.boxplot(showfliers=False)

ax = results_accuracy.boxplot()
ax.set_ylim([0.9,1])
results_accuracy.hist(density=True)
from scipy import stats
alpha = 0.05
s, p = stats.normaltest(results_control_accuracy)
if p<alpha:
    print('Control data is not normal')
else:
    print('Control data is normal')
    
s,p = stats.normaltest(results_experimental_accuracy)
if p<alpha:
    print('Experimental data is not normal')
else:
    print('Experimental data is normal')
    

s, p = stats.wilcoxon(results_control_accuracy[0], results_experimental_accuracy[0])

if p < 0.05:
  print('null hypothesis rejected, significant difference between the data-sets')
else:
  print('null hypothesis accepted, no significant difference between the data-sets')