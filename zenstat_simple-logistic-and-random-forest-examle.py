# On the string of experiments I have been doing, this one is inspired from @anfro18 

# so please let me know your thoughts 

# I am trying to use different classification techniques 

#Done - 1. Logistic Regression  2. Random Forest  

#To be Added -  3. Neural Net 4. Keras/Tener Flow  

import numpy as np  

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dt = pd.read_csv("../input/Dataset_spine.csv")

#see few top rows 

dt.head()
#drop last column

dt.drop('Unnamed: 13', axis=1, inplace=True)

# Convert class attriutes of abnormal and normal to integer values of 0 and 1,using get_dummies

# please note abnormal is 1 here  

dt = pd.concat([dt, pd.get_dummies(dt['Class_att'])], axis=1)

# Drop unnecessary label column in place. 

dt.drop(['Class_att','Normal'], axis=1, inplace=True)

# Rename columns

dt.columns= ['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt',

'thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','Class_cat']

#see top 10 rows

dt.head(10)
#see all variables

dt.describe()
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np 

import pickle 

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split

#class distribution of dependent variable 

sns.countplot(x = "Class_cat", data = dt)

plt.title('Class Variable Distribution')

plt.show()
#List of all columns [['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis','pelvic_slope','Direct_tilt',

#'thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope']]

X = dt[['pelvic_tilt', 'cervical_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius','Direct_tilt',

'thoracic_slope','sacrum_angle','scoliosis_slope', 'degree_spondylolisthesis']]

# Removed three variables pelvic_incidence, pelvic_tilt and sacral_slope as their p-value >.05

# this led to the reduction in overall fit by ~2%. 

y = dt["Class_cat"]

X['intercept'] = 1.0  # so we don't need to use sm.add_constant every time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

#train the model on Training dataset

model = sm.Logit(y_train, X_train)

result = model.fit()
# model results 

result.summary2()
#see the performance 

def logPredict(modelParams, X):  

    probabilities = modelParams.predict(X)

    return [1 if x >= 0.5 else 0 for x in probabilities]

predictions = logPredict(result, X_test)

accuracy = np.mean(predictions == y_test)

#print ('Variable List \n')

#print(X_test.columns.values)

print ('accuracy = {0}%'.format(accuracy*100))

#['lumbar_lordosis_angle' 'pelvic_radius' 'Direct_tilt' 'thoracic_slope', 'cervical_tilt' 'sacrum_angle' 'scoliosis_slope']

#accuracy = 77.06422018348624%

#['lumbar_lordosis_angle' 'pelvic_radius' 'Direct_tilt' 'thoracic_slope', 'sacrum_angle' 'scoliosis_slope']

#accuracy = 72.47706422018348%

#['pelvic_radius' 'Direct_tilt' 'thoracic_slope' 'sacrum_angle', 'scoliosis_slope']

#accuracy = 72.47706422018348%, lumbar_lordosis_angle looks insignificant since we have same accuracy on test set 

#['pelvic_incidence' 'pelvic_tilt' 'sacral_slope' 'pelvic_radius', 'Direct_tilt' 'thoracic_slope' 'sacrum_angle' 'scoliosis_slope']

#accuracy = 73.39449541284404%, but failed to converge 

#['cervical_tilt' 'sacral_slope' 'pelvic_radius' 'Direct_tilt', 'thoracic_slope' 'sacrum_angle' 'scoliosis_slope']

#accuracy = 70.64220183486239%, adding cervical tilt helped in first iteration but it looks as if this varaible has significant when used with other vars 

#['pelvic_tilt' 'cervical_tilt' 'sacral_slope' 'pelvic_radius', 'Direct_tilt' 'thoracic_slope' 'sacrum_angle' 'scoliosis_slope']

#accuracy = 74.31192660550458%, pelvic incidence and pelvic tiltseems to be corelated, removed pelvic incidence 

#['pelvic_tilt' 'cervical_tilt' 'lumbar_lordosis_angle' 'sacral_slope',  'pelvic_radius' 'Direct_tilt' 'thoracic_slope' 'sacrum_angle'

# 'scoliosis_slope']

#accuracy = 76.14678899082568%, whoa, added lumbar_lordosis_angle and we reached at accuracy of 76.14678899082568% without convergence failed error 

#['pelvic_tilt' 'cervical_tilt' 'lumbar_lordosis_angle' 'sacral_slope',  'pelvic_radius' 'Direct_tilt' 'thoracic_slope' 'sacrum_angle'

# 'scoliosis_slope' 'degree_spondylolisthesis']

#accuracy = 83.4862385321101%, added degree_spondylolisthesis, the variable shows as insiignificant

# but has improved the accuracy drastically ,Error : Current function value: inf, Psudo R squared - inf  

#['pelvic_tilt' 'cervical_tilt' 'lumbar_lordosis_angle' 'sacral_slope', 'pelvic_radius' 'Direct_tilt' 'thoracic_slope' 'sacrum_angle'

# 'scoliosis_slope' 'degree_spondylolisthesis']

#accuracy = 81.65137614678899%
#Adding Random Forest technique

from sklearn.ensemble import RandomForestClassifier

#Techniquely, the more trees we use, the precise outcome we would get

x_train2,x_test2,y_train2,y_test2 = train_test_split(X,y,test_size=0.3,random_state=42)



#Generate forests containing 10(default), 50, 100 trees

n_trees = [10, 30,  50, 100]

for i in n_trees:

    mdl = RandomForestClassifier(n_estimators=i)

    mdl.fit(x_train2,y_train2)

    pred = mdl.predict(x_test2)

    

    print('number of trees: {}'.format(i))

    #Each time of prediction,the accuracy is measured

    correct_pred = 0

    for j,k in zip(y_test2,pred):

        if j == k:

            correct_pred += 1

    print('accuracy: {}'.format(correct_pred/len(y_test2) *100))