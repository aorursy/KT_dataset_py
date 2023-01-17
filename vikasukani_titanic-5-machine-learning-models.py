import numpy as np # Linear Algebra



import pandas as pd # Dataset related Filtering



import seaborn as sns # Beautiful Graphs

sns.set_style('dark') # Set graph styles to 'dark'



import matplotlib.pyplot as plt # Normal ploating graphs

# show graphs in this notebook only 

%matplotlib inline



import plotly.express as px # For interactive plots





# ignore  the warning

import warnings  

warnings.filterwarnings('ignore') 



# Read Train.csv File 

trainDF = pd.read_csv('./../input/titanic/train.csv')



# show first five rows from training dataset

trainDF.head()
# Read Test.csv File 

testDF = pd.read_csv('./../input/titanic/test.csv')



#  show 5 rows

testDF.head()
# Print 5 Rows

testDF.head() # train() for last 5 rows
def show_shape(train, test):

    """ 

    display the shape of train and test DF 

    

    """   

    print(" Shape of Training DF", train.shape)

    print("")

    print(" Shape of Testing DF", test.shape)
#  to know shape of the training and testing data

show_shape(trainDF, testDF)
# Create an function to display the information of our train and test dataset. Function can be called multiple time in this notebook.

def show_info(train, test):

    """ 

    display the Information of train and test DF 

    

    """

    

    print("Information of Training DF"+ "-"*10)

    print(train.info())

    print("")

    print("")

    print("")

    print("Information of Testing DF"+ "-"*10)

    print(test.info())
show_info(trainDF, testDF)
removedFeatures = ['Name', 'Ticket', 'Cabin']



trainDF = trainDF.drop(removedFeatures, axis=1) # remove from train DF

testDF = testDF.drop(removedFeatures, axis=1) # remove from test DF



trainDF.head()
# Age Feature



trainDF['Age'] = trainDF['Age'].fillna(trainDF['Age'].mean()) # fill for train DF

testDF['Age'] = testDF['Age'].fillna(testDF['Age'].mean()) # fill for test DF
trainDF['Embarked'].value_counts() # Group Wise count records
# Fill to Embarked column NA with S

 

trainDF['Embarked'] = trainDF['Embarked'].fillna('S') # for train DF only

# show info of train and test data set by calling function



show_info(trainDF, testDF)
# Show servived graph

 



# Plot Counts for Each survived groupby counts

fig = px.bar(trainDF.Survived.value_counts())



fig.show()
 



# Plot Counts for Each survived groupby counts

fig = px.bar(trainDF.groupby(['Survived']).count())



fig.show()
fig = px.histogram(trainDF, x='Survived', y='Pclass', color='Pclass');

fig.show()
 

sns.catplot(x="Pclass", col="Survived", data=trainDF, kind="count");



plt.show()
fig = px.histogram(trainDF, x='Pclass', y= 'Survived', color='Pclass', )

fig.show()
#  Pclass wise survived graph 





plt.figure(figsize=(10, 7))



sns.barplot(x= 'Pclass', y='Survived', data=trainDF)

plt.title("Pclass wise survived ")

plt.show()
# Gender wise Survived graph



fig = px.bar(trainDF, x='Sex', y='Survived', color='Sex')

fig.show()

 
# Parch and Survived Bar graph

 

plt.figure(figsize=(10, 7))



sns.barplot(x = 'Parch', y= 'Survived', data= trainDF)

plt.title("Parch and Survived Graph")



plt.show()
# Embarked and Survived bar Graph

plt.figure(figsize=(10, 7))



sns.barplot(x= 'Embarked', y = 'Survived', data= trainDF)

plt.title("Embarked and Survived Graph")



plt.show()
plt.figure(figsize=(10, 5))

sns.distplot(trainDF.Fare)

plt.title('Distribution of Fares')

plt.show()
# heatmap show

plt.figure(figsize=(10, 7))

sns.heatmap(trainDF.corr(), cmap='Greens', linewidths=1, annot=True, fmt='.1f')



fig=plt.gcf()

plt.show()
# show the info

show_info(trainDF, testDF)
# Fill na with median for Fare feature



testDF["Fare"] = testDF["Fare"].fillna(testDF["Fare"].mean()) # for test DF only
# Convert sex object values to numeric male=1 and female=0, for both train and test DF



trainDF['Sex'] = trainDF['Sex'].replace({'male': 0, 'female': 1})

testDF['Sex'] = testDF['Sex'].replace({'male': 0, 'female': 1})

 
# count values for Embarked

print(testDF['Embarked'].value_counts())

print(trainDF['Embarked'].value_counts())

#  Now, Replace with alphabets to Numbers, for both train and test DF



trainDF['Embarked'] = trainDF['Embarked'].replace({'C': 1, 'S':2, 'Q': 3})

testDF['Embarked'] = testDF['Embarked'].replace({'C': 1, 'S': 2, 'Q': 3})
print(trainDF.head())

print(testDF.head())
# Load Accuracy

from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import confusion_matrix
# Set Prediction value



X_train = trainDF.drop(['PassengerId', 'Survived'], axis=1)

y_train = trainDF['Survived']

X_test = testDF.drop(['PassengerId'], axis=1)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

# Load Model

from sklearn.tree import DecisionTreeClassifier



model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# To predict our model



pred = model.predict(X_test)

pred.shape
# show prediction



accu = model.score(X_train, y_train) # model accuracy

print( "Model Prediction Score", (accu * 100).round(2))

dict = {

    'PassengerId' : testDF['PassengerId'],

    'Survived' : pred

}



new_submission = pd.DataFrame(dict, )

new_submission.shape
# Generate Submission File

# new_submission.to_csv('./my_new_submission.csv', index=False)

# print("Submission Successfully Saved...")
# Import other Models Classes



from sklearn.tree import DecisionTreeClassifier



from sklearn.ensemble import RandomForestClassifier



from sklearn.linear_model import LogisticRegression



from sklearn.svm import SVC, LinearSVC



from sklearn.naive_bayes import GaussianNB



from sklearn.linear_model import SGDClassifier



from sklearn.neighbors import KNeighborsClassifier
def model_wise_predict(models):

    """ 

    Model Predictions

    

    """

    ans_score = []

    for mdl, filename in models:

        m = mdl

        m.fit(X_train, y_train)

        pred = m.predict(X_test)

        m_accuracy = m.score(X_train, y_train)

        ans_score.append((m_accuracy*100).round(2))

        

        dict = {

            'PassengerId' : testDF['PassengerId'],

            'Survived' : pred

        }

        new_submission = pd.DataFrame(dict, )

        

#         Uncomment this line if you want to generate all the csv file for all of the models.

#         new_submission.to_csv(filename, index=False)

        

        

    return ans_score
#  Using DecisionTreeClassifier Model



#  make list of Models

models = [

    (RandomForestClassifier(n_estimators=300, max_depth=20, random_state=5), 'DTC_submission.csv'),

    (RandomForestClassifier(), 'RFC_submission.csv'),

    (LogisticRegression(), 'LR_submission.csv'),

    (LinearSVC(), 'SVC_submission.csv'),

    (GaussianNB(), 'GNB_submission.csv'),

    (SGDClassifier(), 'SGD_submission.csv'),

    (KNeighborsClassifier(), 'KNC_submission.csv')

]



data = model_wise_predict(models)

print("scores are", data)


list_model_name = [

    'DecisionTreeClassifier',

    'RandomForestClassifier',

    'LogisticRegression', 

    'LinearSVC',

    'GaussianNB',

    'SGDClassifier', 

    'KNeighborsClassifier'

]

print(X_train, y_train)
# Customize Model

# TEST



# HYPER TUNNING -------------------------------------------------------------- Start



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



rfc_model = RandomForestClassifier(random_state=35)



rfc_params_grid = {

    'n_estimators' : [600,750,800,850],

    'max_depth' : [7],

    'max_features': [5],

    'min_samples_leaf' : [3],

    'min_samples_split' : [4, 6 ,9],

    'criterion': ["gini", "entropy"]

}



# params = {

#     'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],

#     'n_estimators':[100,250,500,750,1000,1250,1500,1750],

#     'max_depth': np.random.randint(1, (len(train.columns)*.85),20),

#     'max_features': np.random.randint(1, len(train.columns),20),

#     'min_samples_split':[2,4,6,8,10,20,40,60,100], 

#     'min_samples_leaf':[1,3,5,7,9],

#     'criterion': ["gini", "entropy"]

# }



gscv_random_classifier = GridSearchCV(estimator = rfc_model, param_grid = rfc_params_grid, cv = 5 , n_jobs = -1, verbose = 5)

# gscv_random_classifier = RandomizedSearchCV(rfc_model, rfc_params_grid, cv = 5, n_jobs = -1, verbose = 5)



gscv_random_classifier.fit(X_train, y_train)



pred = gscv_random_classifier.predict(X_test)



print("--------------- START ---------------")



# print(accuracy_score(y_test, pred))

print(gscv_random_classifier.best_estimator_)

print(gscv_random_classifier.best_score_)

print(gscv_random_classifier.best_params_)



print("--------------- OVER ---------------")



# HYPER TUNNING -------------------------------------------------------------- END
#  DOWNLOAD SUBMISSION



# Submission FILE EXPORTING  -------------------------------------------------------------- Start

# m = RandomForestClassifier(criterion='entropy', max_depth=7, min_samples_split=4,

#                        n_estimators=750, random_state=42)



m = RandomForestClassifier(max_depth=7, max_features=5, min_samples_leaf=3,

                       min_samples_split=4, n_estimators=600, random_state=35)



m.fit(X_train, y_train)

pred = m.predict(X_test)



print("Acc: ", m.score(X_train, y_train))



dict = {

    'PassengerId' : testDF['PassengerId'],

    'Survived' : pred

}



new_submission = pd.DataFrame(dict, )

new_submission.to_csv('Random-Forest-GSCV-Hyper Tunning.csv', index=False)



# Submission FILE EXPORTING  -------------------------------------------------------------- END



modelDF = pd.DataFrame({"Model_Name" : list_model_name, "Pred_Score": data})

modelDF.sort_values(by='Pred_Score', ascending=False)

modelDF