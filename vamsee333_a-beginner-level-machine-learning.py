import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt


data = pd.read_csv("../input/Admission_Predict.csv")
data.head(3)
## remove un necessary columns - as Serial No doenst have any impact of results

data.drop(['Serial No.'],inplace=True,axis=1)
data.head(2)
## check the correlation between data types
data.corr()
## here the Target is Chance of Admit 

## so now he have to understand the percentage of correlation between available features to target

## lets visualize the relation using heat map

sns.heatmap(data.corr(),annot=True)
## the CGPA stand first followed by GRE Score and TOEFL score

## lets understand each data in detail

## 1.CGPA

data['CGPA'].hist()
## the distribution is symetric here for CGPA

## 2.GRE Score

x=sns.barplot(y='Chance of Admit ',x='GRE Score',data=data)

plt.gcf().set_size_inches(15,10)
## from above we can observe in most of cases with the raise in GRE score chance of admit is increasing

## same way lets check for TOEFL

sns.kdeplot(data['TOEFL Score'],data['Chance of Admit '],shade=True)
data.columns
## you dont have any set of rules in choosing the plot type for data visualization

## the plot that best describes the relation and infromation best suits for visualization

## you can try with any no of plots you like

## lets also see the regression plot to validate the features

sns.regplot(x='GRE Score',y='Chance of Admit ',data=data)
## the above plot clearly says the relationship,as the data points are almost close to the regression line in most of the cases

## box plot is the best plot to find out the outliers

sns.boxplot(y='CGPA',data=data,showmeans=True,meanline=True)

plt.gcf().set_size_inches(5,5)
## the above plot says only one outlier avaliable and inner quartile range gives 50% of data -most of the cgpa in inner quartile range varies in b/w 8.2 to 9.1

## Now bivariate analysis 

## here we are going to comare two features
sns.boxplot(x='University Rating',y='GRE Score',data=data)
## it shows many box plots 

## now lets understand the above figure

## here we can see outliers in second,fourth and fifth box plots

## second box says - student with score in 330 - 340 range is expecting university of rating 2 - but why :) ?

## in 4 and 5,students with range of 290 - 310 are expecting top rated universities 
## can try the same for CGPA

sns.boxplot(y='CGPA',x='University Rating',data=data)
## now its time to compare multiple feature values

## note the word - lmplot, you many come across this many times
data.columns
sns.lmplot(x='Chance of Admit ',y='GRE Score',col='University Rating',data=data)
## its (facetgrid - means a combination of (axes - means a single plot))
## okay now ,we understood our data using the heat map ,correlation and other data visualization

## let me draw a heat map for reference

sns.heatmap(data.corr(),annot=True)
## from above grap i can see all the features having good correlation- all features with more than 65%

## so we can try with different combination of features 

## fisrtly im going to select GRE,TOEFL,CGPA as features which are top 3 and Chance of Admit is our Target



features=data[['GRE Score','TOEFL Score','CGPA']]

target=data[['Chance of Admit ']]
features.head()
target.head()
## lets import ML libraries

## sklearn library holds all the algorithms that is used to predict the values

## though we are just reusing the in built libraries,it is very important to learn algorithms and the logics behind it

from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(features,target,random_state=1,test_size=0.2)
## what did i do-

## I've imported a method called train_test_split,as name says its going to split the data into train and test part

## lets see the data one by one
X_train.head()
X_val.head()
y_train.head()
y_val.head()
## now with train data we are going to make our model learn our data 

## like i mentioned before ,many algorithms are available in sklearn library to build our model

## lets use linear regression mmodel

from sklearn.linear_model import LinearRegression
## here model is studing our train data

student_model=LinearRegression()

student_model.fit(X_train,y_train)
## to check the score of our current train data

student_model.score(X_train,y_train)
## its 80% which is pretty good to build the model
## here we have predited the score by giving the X_val(sample output - that is chances of getting admission as input)

y_pred=student_model.predict(X_val)
## now we are going to compare how close the above predicted value is with the y_val

## for this we have set of inbuilt functions that can be imported

from sklearn.metrics import r2_score
accuracy_result_LR=round(r2_score(y_pred,y_val)*100,2)
## its 69 which is a good model,but we can try for better accuracy if possible

accuracy_result_LR
from sklearn.tree import DecisionTreeRegressor
DTR_model=DecisionTreeRegressor()
DTR_model.fit(X_train,y_train)
y_pred=DTR_model.predict(X_val)

accuracy_result_DTR=round(r2_score(y_pred,y_val)*100,2)

accuracy_result_DTR
## the decision tree model is giving very low prediction results, so lets try with new feature combination 
## here ive included all features

## for this data we havent encountered any feature with less correlation, but you ll going to deal with lot of data that gives very bad correlation

features2=data.drop(['Chance of Admit '],axis=1)

target2=data['Chance of Admit ']
X_train2,X_val2,y_train2,y_val2=train_test_split(features2,target2,random_state=1,test_size=0.2)
lin_model=LinearRegression()
lin_model.fit(X_train2,y_train2)
y_pred=lin_model.predict(X_val2)

accuracy_per=round(r2_score(y_pred,y_val2)*100,2)

accuracy_per
lin_model.score(X_train2,y_train2)
## this is a good considerable accuracy score

print(f'the accuracy percentage for our model is {accuracy_per}')
pd.to_pickle(lin_model,'chance_prediction.pickle')
## why do we pickle?

## when we deal with large data frames ,its going to take long time to build and train a model

## so on pickling a model ,we dont need to build and train a model 
## this is the way to extrat the pickle file and reuse it

## my_model=pd.read_pickle('chance_prediction.pickle')
##we have built our model sucessfully - when any user inputs are to be given , uncomment the below code and pass the input

## GRE=int(input('enter GRE score ,360 is max score - '))

## TOEFL=int(input('enter TOEFL score ,120 is max score -'))

## required_university_rating= int(input('enter university rating in 1-5 range -'))

## SOP=float(input('enter sop score in 1-5 range-'))

## LOR=float(input('enter LOR score in 1-5 range-'))

## CGPA=float(input('enter CGPA in 1-10 range-'))

## Research=int(input('enter Research score 0-if no and 1-if yes-'))
## inputs=[GRE,TOEFL,required_university_rating,SOP,LOR,CGPA,Research]
## this will be the result predition

## result=my_model.predict([inputs])

## print(f'so,the probability of you getting desired University is {round(result[0]*100,2)}')