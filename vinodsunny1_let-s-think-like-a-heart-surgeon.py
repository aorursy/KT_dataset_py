import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import plotly.express as ex

sns.set_style('darkgrid')

from IPython.display import Image
Image('../input/heart-image/heart.jpg')
heart = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

print("shape of th dataframe  : ",heart.shape)
# renaming columns for readablity purpose

heart.columns = ['age','sex','chest_pain','person_blood_pressure','cholestrol','fasting_blood_sugar','resting_ecg','max_hrate','exang'

                ,'oldpeak','slope','ca','thal','target']

df  = heart.copy()
# We Don't Have any Null value's here .

heart.isnull().sum()
# top 5 rows

heart.head(5)
heart.info()
sns.countplot(heart['target'])

plt.title("Target Observation ")

plt.show()
ones  = heart[heart['target']==1] # people with heart disease

zeros = heart[heart['target']==0] # people without heart disease
Image('../input/heartgif/heart.gif')
plt.rcParams['figure.figsize'] = (14,5)

plt.subplot(1,2,1)

sns.kdeplot(heart['age'][heart.target == 1],shade = True,color = "red")

plt.title('People With Heart Disease ')

plt.xlabel('Age distribution of people with heart disease ')

plt.subplot(1,2,2)

sns.kdeplot(heart['age'][heart.target == 0],shade = True,color = "green")

plt.title('People Without  Heart Disease ')

plt.xlabel('Age distribution of people with heart disease ')

pd.crosstab(heart.age,heart.target).plot(kind = 'bar',figsize = (15,7))

plt.title("Age v/s target")

plt.ylabel("Frequency")
# sex 1 being male ,0 being female

heart['sex'][heart['sex']==1]='male'

heart['sex'][heart['sex']==0]='female'



# renaming chest pain column

heart['chest_pain'][heart['chest_pain']==0]= 'typical angina'

heart['chest_pain'][heart['chest_pain']==1]= 'atypical angina'

heart['chest_pain'][heart['chest_pain']==2]= 'non-anginal pain'

heart['chest_pain'][heart['chest_pain']==3]= 'asymptomatic'





# renaming fasting_blood_sugar 

heart["fasting_blood_sugar"][heart["fasting_blood_sugar"]==1]= 'higher than 120mg/ml'

heart["fasting_blood_sugar"][heart["fasting_blood_sugar"]==0]= 'lower than 120mg/ml'



#renaming  slope 

heart['slope'][heart['slope']==0]= 'upsloping'

heart['slope'][heart['slope']==1]= 'flat'

heart['slope'][heart['slope']==2]= 'downsloping'

heart['slope'] = heart['slope'].astype('object')
plt.rcParams['figure.figsize'] = (8,3)

sns.countplot(heart['sex'][heart.target == 1],palette  = 'BrBG')

plt.ylabel("Heart Diesease frequency")
sns.countplot(heart['sex'],hue = heart['target'])

plt.title("Gender V/S Heart Disease")
sns.boxplot(heart['target'],heart['person_blood_pressure'])

plt.title("person_blood_pressure v/s target ")
plt.rcParams['figure.figsize'] = (8,3)

sns.boxenplot(heart['target'],heart['cholestrol'])

plt.title("cholestrol v/s target ")
pd.crosstab(heart['fasting_blood_sugar'],heart['target']).plot(kind = 'bar',figsize=(10,4),color=['#FFC300','#581845'])

plt.title("fasting_blood_sugar v/s target")
pd.crosstab(heart["chest_pain"],heart['target']).plot(kind = 'bar',figsize = (16,4),color = ["red","green"])

plt.title("Chest pain v/s Target")  
# From below chart now ,it's clearly confirmed that male's suffer more from heart disease than female's .

pd.pivot_table(heart,index = ["sex","chest_pain"],values = ["target"],aggfunc = "count").plot(kind = 'bar',figsize = (16,4),colormap = 'PRGn')

plt.title("Sex v/s ChestPain")
# Relationship between Heart v/s Age .

plt.rcParams['figure.figsize'] = (12,6)

# from below scatter we can conclude that people with heart disease has Higher Heart rate compared to the people who don't have . 

plt.scatter(x = heart.age[(heart.target == 1)] , y = heart.max_hrate[(heart.target == 1)])

plt.scatter(x = heart.age[(heart.target == 0)] , y = heart.max_hrate[(heart.target == 0)])

plt.legend([ "with heart disease","without heart disease"])

plt.title("Relationship between Heart Rate with Age ")

plt.xlabel("age")

plt.ylabel("Max Heart")
pd.crosstab(heart.exang,heart.target).plot(kind = 'bar',colormap = 'Wistia')

plt.title("Relationship of excercise  Angina with Target ")

plt.ylabel("Target Frequency ")
# From the below chart we can conclude that ,

# people with heart disease have more blood pressure compared to people who do not have .

sns.boxplot(heart["target"],heart["person_blood_pressure"],hue = heart["sex"],palette = 'viridis')

plt.title("Blood Pressure v/s Target")
# From the below chart we conclude that people with or without heart disease tend to have  very less resting ecg value = 2 , 

# but people with heart disease tend to have  very high value for resting ecg = 0 

pd.crosstab(heart['target'], heart['resting_ecg']).plot(kind = 'bar')
plt.rcParams['figure.figsize'] = (12,4)

# relationship Between Major vessel's with target

# from below chart we can conclude that poeple with heart disease have more major cell type = 0,

# compared to people who do not have heart disease . 

pd.crosstab(heart['target'],heart['ca']).plot(kind = 'bar',colormap = 'Blues_r')

plt.title("Relationship Between Major vessel's (CA) with Target")

plt.ylabel("CA Freaquency")
# Relationship Between Slope v/s Target .



# 0: Upsloping: better heart rate with excercise (uncommon)

# 1: Flatsloping: minimal change (typical healthy heart)

# 2: Downslopins: signs of unhealthy heart



# From Below chart it is clearly visible that people with heart diesease Have More value For Downsloping 

# (which indicates sign of unhealthy heart ) .



pd.crosstab(heart['target'],heart['slope']).plot(kind = 'bar',color = ['r','g','y'])

plt.title("Relationship Between Slope v/s Target")
# Relationship Between thal (thalium stress result) with Target .

# thal :  is Blood Disorder .

# 0,1  : normal

# 2    : fixed defect: used to be defect but ok now

# 3    :  reversable defect: no proper blood movement when excercising



# it is clearly visible that People with heart disease are more likely to not suffer from this Blood disorder (Thalessemia) 

# compared with people who are not having heart disease . 



sns.boxenplot(heart['target'],heart['thal'])

plt.title("Relationship Between thal (thalium stress result) with Target ")
# Let's See The Correlation Among The Features .



# Below chart is used to visualize how one feature is correlated with every other Features Present in the dataset .

# if we have two highly correlated features then we will consider only one of them to avoid overfitting .



# since in our Dataset There is now two  features which are highly correlated ,

# hence we have consider all the features for training our Model .





plt.rcParams['figure.figsize'] = (18, 10)

sns.heatmap(df.corr(),annot = True ,cmap = 'rainbow_r',annot_kws = {"Size":14})

plt.title( "Chart Shows Correlation Among Features   : ")
# Converting Data Type From Int to Object .



heart['age']    = heart['age'].astype(int)

heart['resting_ecg'] = heart['resting_ecg'].astype('str')

heart['exang'] = heart['exang'].astype('str')

heart['thal']  = heart['thal'].astype('str')

heart['ca']    = heart['ca'].astype('str')
heart.info()
# Model Building 
x = heart.drop(['target'],axis = 1)

y = heart.target

print(x.shape)
x.columns
x = pd.get_dummies(x,drop_first = True) # for converting categorical to Numerical value .

print(x.shape)
x.columns
x.head() # dataset after applying get dummies function
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_auc_score,roc_curve



xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 33)

print("Train Set Size : ",xtrain.shape)

print("Train Target Set Size : ",ytrain.shape)

print("Test  Set Size : ",xtest.shape)

print("Test  Target Set Size : ",ytest.shape)
# Applying Scaling Standardiztion to all of the features in order to bring them into common scale .

# Standardiztion : is preferred when most of the featues are not following gaussian distribution . 



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xtrain = pd.DataFrame(sc.fit_transform(xtrain))

xtest  = pd.DataFrame(sc.fit_transform(xtest))
# Importing GridSearchCv from sklearn in order to find out the optimal Parameter for given Algorithm

# that gives best result .



from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 33 )



# Setting Parameters for Logistic Regression . 



params = {    # Regularization Params

             'penalty' : ['l1','l2','elasticnet'],

              # Lambda Value 

             'C' : [0.01,0.1,1,10,100]

         }



log_reg = GridSearchCV(lr,param_grid = params,cv = 10)

log_reg.fit(xtrain,ytrain)

log_reg.best_params_
# Make Prediction of test data 

ypred = log_reg.predict(xtest)

print(classification_report(ytest,ypred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(ytest,ypred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for Logistic Regression  Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = log_reg.predict_proba(xtest)[:,1]



# display auc value for log_reg

auc_log_reg = roc_auc_score(ytest,ypred)

print("roc_auc_score value for log reg is : ",roc_auc_score(ytest,ypred))



# Create true and false positive rates

fpr_log_reg,tpr_log_reg,thershold_log_reg_model = roc_curve(ytest,y_proba)

plt.plot(fpr_log_reg,tpr_log_reg)

plt.plot([0,1],ls='--')

#plt.plot([0,0],[1,0],c='.5')

#plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For Logistic Regregression')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.tree import DecisionTreeClassifier 



dt = DecisionTreeClassifier(random_state = 33)





# Setting Parameters for DecisionTreeClassifier . 



params = {  

             'criterion'    : ["gini", "entropy"],

             'max_features' : ["auto", "sqrt", "log2"],

              'min_samples_split' :[i for i in range(4,16)],

              'min_samples_leaf' : [i for i in range(4,16)]

         }



dt_clf = GridSearchCV(dt,param_grid = params,cv = 10)

dt_clf.fit(xtrain,ytrain)

dt_clf.best_params_
# Make Prediction of test data 

ypred = dt_clf.predict(xtest)

print(classification_report(ytest,ypred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(ytest,ypred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for DecisionTreeClassifier   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

plt.show()
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = dt_clf.predict_proba(xtest)[:,1]



dt_clf_auc_score = roc_auc_score(ytest,ypred)

# display auc value for DecisionTreeClassifier

print("roc_auc_score value for log reg is : ",roc_auc_score(ytest,ypred))



# Create true and false positive rates

fpr_dt_clf,tpr_dt_clf,thershold_dt_clf_model = roc_curve(ytest,y_proba)

plt.plot(fpr_dt_clf,tpr_dt_clf)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For DecisionTreeClassifier ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators = 150,min_samples_split = 20,min_samples_leaf = 5,random_state = 33)

rf_clf.fit(xtrain,ytrain)

yperd = rf_clf.predict(xtest)
# Make Prediction of test data 

ypred = rf_clf.predict(xtest)

print(classification_report(ytest,ypred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(ytest,ypred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for RandomForestClassifier   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = dt_clf.predict_proba(xtest)[:,1]



rf_auc_score = roc_auc_score(ytest,ypred)



# display auc value for RandomForestClassifier

print("roc_auc_score value for log reg is : ",roc_auc_score(ytest,ypred))



# Create true and false positive rates

fpr_rf_clf,tpr_rf_clf,thershold_rf_clf_model = roc_curve(ytest,y_proba)

plt.plot(fpr_rf_clf,tpr_rf_clf)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For RandomForestClassifier ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_jobs = -1)



# set params



params = {

             "n_neighbors" : [i for i in range(15)],

               'p' : [1,2] ,

              'leaf_size' : [i for i in range(15)],

               

          }

knn = GridSearchCV(knn,param_grid = params, cv = 5)

knn.fit(xtrain,ytrain)

knn.best_params_
# Make Prediction of test data 

ypred = knn.predict(xtest)

print(classification_report(ytest,ypred))
plt.rcParams['figure.figsize'] = (6,4)

class_names = [1,0]

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks,class_names)

plt.yticks(tick_marks,class_names)



#create a heat map

sns.heatmap(pd.DataFrame(confusion_matrix(ytest,ypred)), annot = True, cmap = 'BuGn_r',

           fmt = 'g')

plt.tight_layout()

plt.title('Confusion matrix for KNN Algorithm   Model', y = 1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
plt.rcParams['figure.figsize'] = (10,6)



# Get predicted probabilites from the model

y_proba = knn.predict_proba(xtest)[:,1]



knn_auc_score = roc_auc_score(ytest,ypred)





# display auc value for KNN Algorithm

print("roc_auc_score value for log reg is : ",roc_auc_score(ytest,ypred))



# Create true and false positive rates

fpr_KNN,tpr_KNN,thershold_KNN_model = roc_curve(ytest,y_proba)

plt.plot(fpr_KNN,tpr_KNN)

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.title('Reciever Operating Characterstic For KNN Algorithm ')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
plt.figure(figsize=(10,6))

plt.title('Reciever Operating Characterstic Curve')

plt.plot(fpr_log_reg,tpr_log_reg,label='LogisticRegression')

plt.plot(fpr_dt_clf,tpr_dt_clf,label='DecisionTreeClassifier')

plt.plot(fpr_rf_clf,tpr_rf_clf,label='RandomForestClassifier')

plt.plot(fpr_KNN,tpr_KNN,label='KNearestNeighbors ')

plt.plot([0,1],ls='--')

plt.plot([0,0],[1,0],c='.5')

plt.plot([1,1],c='.5')

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.legend()

plt.show()

print("Area Under Curve Score values for Different algorithms : ")

print("LogisticRegression          : ",auc_log_reg)

print("DecisionTreeClassfier       : ",dt_clf_auc_score)

print("RandomForest Classifier     : ",rf_auc_score)

print("KnearestNeighborsClassifier : ",knn_auc_score)