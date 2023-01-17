import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn.preprocessing as skp
#Reading the data
data=pd.read_csv('../input/data.csv')
data.head()
#Description of data
data.describe(include='all')
#Reviewing of data types
data.dtypes
#Converting the numbered categorical data into object type
data['Year']=data['Year'].astype('category')
data['Engine Fuel Type']=data['Engine Fuel Type'].astype('category')
data['Number of Doors']=data['Number of Doors'].astype('category')
data['Engine Cylinders']=data['Engine Cylinders'].astype('category')
data.describe(include='all')
#Missing value treatment
data['Market Category']=data['Market Category'].fillna('missing')
#There are 3728 missing values are there which is  31.56%
# Hence it would not be a good idea to impute it instead it would be better to treat it as seperate observed category
data.describe(include='all')
data=data.dropna()
data.describe(include='all')
%matplotlib inline
size=(20,9)
plt.subplots(figsize=size)
plt.xticks(rotation=90)
sb.stripplot(x='Market Category',y='MSRP',data=data,size=15,hue='Year')

%matplotlib inline
size=(11,9)
plt.subplots(figsize=size)
plt.xticks(rotation=90)
sb.stripplot(x='Year',y='MSRP',data=data,size=10)
# After Year 2000, there is a sharp increase in the price band
#either the data are missing or there are no data collected for high price band cars before 2001 or there are no cars available at high price
#Outlier analysis
#Though outliers are present in the data, to have the idea of highly priced cars , it would be better to keep the data
a4=(20,10)
plt.figure(figsize=a4)
plt.boxplot(data['Engine HP'], vert=False)
#Correlation matrix between MSRP and other continous variables
cont_data=data[['Engine HP', 'MSRP', 'city mpg','highway MPG','Popularity']]
cat_data=data.drop(['Engine HP', 'MSRP', 'city mpg','highway MPG','Popularity'],axis=1)
cont_data.head()
cat_data.head()
cat_data['MSRP']=data['MSRP']
cat_data.head()
sb.pairplot(cont_data)
#Correlataion matrix between MSRP and other categorical variables
sb.pairplot(cat_data)
#Basic statistis
plt.hist(data['MSRP'])
#getting categorical variables into dummies
data=pd.get_dummies(data,dummy_na=False,columns=['Make','Model','Engine Fuel Type', 'Transmission Type','Market Category','Driven_Wheels','Vehicle Size','Vehicle Style'])

#Splitting Target and Labels data
Labels=np.array(data['MSRP'])
Features=data.drop('MSRP',axis=1)
Features=np.array(Features)

#splitting between Test and Train datasets
import sklearn
from sklearn.model_selection import train_test_split
Train_Features, Test_Features,Train_Labels,Test_Labels=train_test_split(Features,Labels,test_size=0.2,random_state=42)
#Random forest model training
#from sklearn.ensemble import RandomForestRegressor
#RFTree=RandomForestRegressor(n_estimators=500,random_state=42)
#Trainmodel=RFTree.fit(Train_Features,Train_Labels)
#print('Training model done')
#prediction using model
#Pred_MSRP=RFTree.predict(Test_Features)
#errors=abs(Pred_MSRP-Test_Labels)
#print('Mean Absolute Error:', round(np.mean(errors), 2))
#Model Accuracy
#MAPE=100*(errors/Test_Labels)
#Accuracy=100-np.mean(MAPE)
#print('Accuracy:', round(Accuracy,2),'%')
#Cross Validation
#from sklearn.model_selection import cross_val_score
#CV_score=cross_val_score(RFTree,Test_Features,Test_Labels,cv=5)
#print('CV Score:',CV_score)
#Further statistics
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import explained_variance_score
#mean_absolute_error(Test_Labels,Pred_MSRP)
#explained_variance_score(Test_Labels,Pred_MSRP)
#np.sqrt(mean_squared_error(Test_Labels,Pred_MSRP))
#RMSE=np.sqrt(mean_squared_error(Test_Labels,Pred_MSRP))
#print(RMSE/np.mean(Test_Labels))
#with the given statistics and cros vakidation score it is observed that the fitted model suits the data for the prediction.
#Random forest model training
#from sklearn.ensemble import RandomForestRegressor
#RFTree2=RandomForestRegressor(n_estimators=500,random_state=42,min_impurity_split=0.0001)
#Trainmodel=RFTree2.fit(Train_Features,Train_Labels)
#print('Training model done')
#prediction using model
#Pred_MSRP2=RFTree2.predict(Test_Features)
#errors2=abs(Pred_MSRP-Test_Labels)
#print('Mean Absolute Error:', round(np.mean(errors), 2))
#Decision tree learning model
from sklearn.tree import DecisionTreeRegressor
#print("Starts training model")
#Dtree= DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2,
 #                            min_samples_leaf=1)
                             
#Dtree_Model=Dtree.fit(Train_Features,Train_Labels)
#print("Trainng model done")
#prediction using model
#dtree_Pred_MSRP=Dtree.predict(Test_Features)
#dtree_errors=abs(dtree_Pred_MSRP-Test_Labels)
#print('Mean Absolute Error:', round(np.mean(dtree_errors), 2))
#print(Dtree_Model)
##Model Accuracy
#MAPE=100*(dtree_errors/Test_Labels)
#Accuracy=100-np.mean(MAPE)
#print('Accuracy:', round(Accuracy,2),'%')
#from sklearn.metrics import mean_squared_error
#RMSE=np.sqrt(mean_squared_error(Test_Labels,dtree_Pred_MSRP))
#print(RMSE)
#Adaboost regression model 
from sklearn.ensemble import AdaBoostRegressor
print('Training model starts')
ABTree=AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=500,loss='linear', learning_rate=1.0)
ABtreemodel=ABTree.fit(Train_Features,Train_Labels)
print('Training model done')
#prediction using model
ABtree_Pred_MSRP=ABTree.predict(Test_Features)
ABtree_errors=abs(ABtree_Pred_MSRP-Test_Labels)
print('Mean Absolute Error:', round(np.mean(ABtree_errors), 2))
##Model Accuracy
MAPE=100*(ABtree_errors/Test_Labels)
Accuracy=100-np.mean(MAPE)
print('Accuracy:', round(Accuracy,2),'%')





















































































































































