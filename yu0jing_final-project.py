# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from sklearn import datasets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
iris = datasets.load_iris()
x = pd.DataFrame(iris['data'],columns=iris['feature_names'])
print('target_names:' + str(iris['target_names']))
y = pd.DataFrame(iris['target'],columns=['target'])
iris_data = pd.concat([x,y],axis = 1)
iris_data.head(3)
iris['target_names']
target_name = {0:'setosa',1:'versicolor',2:'virginica'}
iris_data['target_name'] = iris_data['target'].map(target_name)
iris_data = iris_data[(iris_data['target_name'] == 'setosa')|(iris_data['target_name'] == 'versicolor')]
iris_data = iris_data[['sepal length (cm)','petal length (cm)','target_name']]
iris_data.head(5)
target_class = {'setosa':1,'versicolor':-1}
iris_data['target_class'] = iris_data['target_name'].map(target_class)
del iris_data['target_name']
iris_data.head(3)
def sign(z):
    if z > 0:
        return 1
    else:
        return-1
w = np.array([0.,0.,0.])
error = 1
iterator = 0
while error !=0:
    error = 0
    for i in range(len(iris_data)):
        x,y = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2])), np.array(iris_data.iloc[i])[2]
        if sign(np.dot(w,x)) !=y:
            print('iterator:'+str(iterator))
            iterator +=1
            error +=1
            sns.lmplot('sepal length (cm)','petal length (cm)',data = iris_data,fit_reg = False, hue = 'target_class')
            
            if w[1] != 0:
                x_last_decision_boundary = np.linspace(0,w[1])
                y_last_decision_boundary = (w[2]/w[1])*x_last_decision_boundary
                plt.plot(x_last_decision_boundary,y_last_decision_boundary,'c--')
            w +=y*x
            print("w:"+str(w))
            print("x:"+str(x))
            
            x_vector = np.linspace(0,x[1])
            y_vector = (x[2]/x[1])*x_vector
            plt.plot(x_vector,y_vector,"b")
            
            x_decision_boundary = np.linspace(-0.5,7)
            y_decision_boundary = (-w[1]/w[2])*x_decision_boundary-(w[0]/w[2])
            plt.plot(x_decision_boundary,y_decision_boundary,'r')
            
            x_decision_boundary_normal_vector = np.linspace(0,w[1])
            y_decision_boundary_normal_vector = (w[2]/w[1])*x_decision_boundary_normal_vector
            plt.plot(x_decision_boundary_normal_vector,y_decision_boundary_normal_vector,'g')
            plt.xlim(-0.5,7.5)
            plt.ylim(5,-3)
            plt.show()
            
            
            

