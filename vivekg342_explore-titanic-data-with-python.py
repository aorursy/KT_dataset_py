import csv as csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt  

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

import scipy.optimize as opt  

from sklearn import metrics, linear_model, tree, ensemble

from sklearn.grid_search import GridSearchCV
# Function to get agggreated values based on a ticket

def get_ticket_aggr(initTrain, initFinal):

    trainFrame = pd.concat([initTrain.drop(['Survived'], axis = 1), initFinal])

    a1 = trainFrame.groupby('Ticket')['PassengerId'].count()

    a2 = initTrain.groupby('Ticket')['Survived'].sum()

    agg = a1.to_frame().join(a2.to_frame()).reset_index()

    agg = agg.rename(columns={'PassengerId': 'TicketCount', 'Survived': 'TicketSurvived'})

    agg.loc[agg.TicketSurvived.isnull(), 'TicketSurvived'] = 1

    agg['TicketSurvived?'] = (agg.TicketSurvived > 1).astype(bool)

    agg['TicketCount'] = agg['TicketCount'].astype(int)

    agg = agg.drop(['TicketSurvived', 'TicketSurvived?'], axis = 1)

    return agg



# Function to get agggreated values based on a surname

def get_surname_aggr(initTrain, initFinal):

    trainFrame = pd.concat([initTrain.drop(['Survived'], axis = 1), initFinal])

    a3 = trainFrame.groupby('surname')['PassengerId'].count()

    a4 = initTrain.groupby('surname')['Survived'].sum()

    agg2 = a3.to_frame().join(a4.to_frame()).reset_index()

    agg2 = agg2.rename(columns={'PassengerId': 'SurCount', 'Survived': 'SurSurvived'})

    agg2.loc[agg2.SurSurvived.isnull(), 'SurSurvived'] = 1

    agg2['SurSurvived?'] = (agg2.SurSurvived > 1).astype(bool)

    agg2['SurCount'] = agg2['SurCount'].astype(int)

    agg2 = agg2.drop(['SurSurvived', 'SurSurvived?'], axis = 1)    

    return agg2



# surname seems to have good correlation with survival

def transform_0(dFrame):

    t1 = dFrame.Name.str.split(',', expand=True)

    t1.columns = ['surname', 'GivenName']

    t1['title'] = t1.GivenName.str.extract('(Mrs|Mr|Miss|Master)', expand=False)

    dFrame = pd.concat([t1, dFrame], axis=1)

    return dFrame



def transform_1(dFrame, catTypes, ticket_agg, surname_agg):

# Extracting features from Name

    dFrame['FamilySize'] = dFrame['SibSp'] + dFrame['Parch']

    dFrame = dFrame.drop(['SibSp', 'Parch'], axis=1) 

    dFrame = dFrame.drop(['PassengerId', 'Name','GivenName', 'Cabin'], axis=1) 

    dFrame['Child'] = dFrame.Age<16

    dFrame['Child'] = dFrame['Child'].astype(bool)

    # Factoring in survivors on the same ticket

    dFrame = pd.merge(dFrame, ticket_agg, how='left', on='Ticket')





    # Factoring in survivors on the same surname

    dFrame = pd.merge(dFrame, surname_agg, how='left', on='surname')

    

    # Fill NA for string fields

    for cat in catTypes:

        dFrame.loc[dFrame[cat].isnull(), cat] = 'NA'



    # Fill fare

    dFrame.loc[dFrame.Fare.isnull(),'Fare'] = dFrame.Fare.median()

    # Filling Age where it is null

    dFrame['AgeIsNull'] = pd.isnull(dFrame.Age).astype(bool)



    dFrame['EmbarkedIsNull'] = pd.isnull(dFrame.Embarked).astype(bool)    

    

    dFrame['Gender'] = dFrame['Sex'].map({'male': 1, 'female': 0}).astype(int)

    median_ages = np.zeros((2,3))

    for i in range(0, 2):

        for j in range(0, 3):

            median_ages[i,j] = dFrame[(dFrame['Gender'] == i) & \

                              (dFrame['Pclass'] == j+1)]['Age'].dropna().median()

    for i in range(0, 2):

        for j in range(0, 3):

            dFrame.loc[ (dFrame.Age.isnull()) & (dFrame.Gender == i) & (dFrame.Pclass == j+1),\

                'Age'] = median_ages[i,j]

    dFrame = dFrame.drop(['Ticket', 'surname', 'Gender'], axis=1)

    dFrame['Age*Class'] = dFrame.Age * dFrame.Pclass

    return dFrame

    

def getColumns(t1, t2):

    #INIT CATEGORY DICTIONARY

    dict = {}

    for cat in catTypes:

        cct = list(t1[cat].unique()) + list(t2[cat].unique())

        dict[cat] = set(cct)

    return dict



# Splits into train and test data set and runs a given grid search on a model

def trainAndTest(clf, test_size=20):

    x_train = Xtrain[:test_size]

    y_train = ytrain[:test_size]

    x_test = Xtrain[test_size:]

    y_test = ytrain[test_size:]

    clf.fit(x_train, y_train)

    print (clf.best_score_, clf.best_params_)

    model = clf.best_estimator_

    predictions = model.predict(x_train)  

    name = type(model).__name__

    print(name)

    print("Train Accuracy {x}, F1 score {f}".format(x=metrics.accuracy_score(predictions, y_train), f=metrics.f1_score(predictions, y_train)))

    predictions_test = model.predict(x_test)  



    accuracy = metrics.accuracy_score(predictions_test, y_test)

    f1_score = metrics.f1_score(predictions_test, y_test)

    print("Test Accuracy {x}, F1 score {f}".format(x=accuracy, f=f1_score))

    return {'accuracy': accuracy, f1_score: f1_score, 'model': model, 'name': name, 'predictions': predictions_test}
# Constants

random_max_depth = 10

test_size = 267



catTypes = ['Pclass', 'Sex', 'title', 'Embarked']



hotTypes = ['Pclass', 'Sex']



initTrain = pd.read_csv('../input/train.csv', header=0)

print("-1) train shape {}".format(initTrain.shape))

initFinal = pd.read_csv('../input/test.csv', header=0)

print("-1) final set shape {}".format(initFinal.shape))



transTrain = transform_0(initTrain)

print("0) train transformed {}".format(transTrain.shape))

transFinal = transform_0(initFinal)

print("0) final transformed {}".format(transFinal.shape))





ticket_agg = get_ticket_aggr(transTrain, transFinal)



final_ticket_agg = get_ticket_aggr(transTrain, transFinal)



sur_agg = get_surname_aggr(transTrain, transFinal)



final_sur_agg = get_surname_aggr(transTrain, transFinal)



transTrain = transform_1(transTrain,  catTypes, ticket_agg, sur_agg)

print("1) train transformed {}".format(transTrain.shape))

transFinal = transform_1(transFinal, catTypes, final_ticket_agg, final_sur_agg)

print("1) final transformed {}".format(transFinal.shape))



catDict = getColumns(transTrain, transFinal)



columns = transTrain.drop(hotTypes+['Survived'], axis=1).columns



# Label encoding for string types

for cat in catTypes:

    lenc = LabelEncoder()

    lenc.fit(list(catDict[cat]))

    transTrain[cat] = lenc.transform(transTrain[cat])

    transFinal[cat] = lenc.transform(transFinal[cat])



    

Xtrain = transTrain.drop(hotTypes+['Survived'], axis=1).values

ytrain = transTrain.Survived.values

Xfinal = transFinal.drop(hotTypes, axis=1).values



# Hot encoding for few features

for cat in hotTypes:

    enc = OneHotEncoder(dtype=np.bool)

    enc.fit(transTrain[cat].reshape(-1,1))

    for r in range(0,enc.n_values_):

        columns = np.append(columns, "{0}_{1}".format(cat, r))  

    train_cat_features = enc.transform(transTrain[cat].reshape(-1,1)).toarray()

    test_cat_features = enc.transform(transFinal[cat].reshape(-1,1)).toarray()    

    Xtrain = np.concatenate([Xtrain,train_cat_features], axis=1)    

    Xfinal = np.concatenate([Xfinal,test_cat_features], axis=1)

    





print("2) training transformed {}".format(Xtrain.shape))

print("2)final set transformed{}".format(Xfinal.shape))
linear_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

decision_tree_param_grid = {'max_depth':list(range(2,20))}

random_param_grid = { 

    'n_estimators': [5, 10, 50, 100],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth':list(range(2,10))

}

classifiers = [

    GridSearchCV(linear_model.LogisticRegression(solver='liblinear', penalty = 'l2'), linear_param_grid),

    GridSearchCV(tree.DecisionTreeClassifier(), decision_tree_param_grid),

    GridSearchCV(ensemble.RandomForestClassifier(), random_param_grid)

]

accuracy = 0

model = None

results = np.array([])

for classifier in classifiers: 

    result = trainAndTest(classifier, test_size)

    results = np.append(results, result)

    if(result['accuracy'] > accuracy):

        accuracy = result['accuracy']

        model = result['model']
model = ensemble.RandomForestClassifier(max_depth=3, max_features='auto', n_estimators=50)

model.fit(Xtrain, ytrain)

yFinal = model.predict(Xfinal) 

output =  pd.DataFrame({'PassengerId': initFinal.PassengerId, 'Survived':yFinal})

output.groupby('Survived').size()
model = linear_model.LogisticRegression(C=1,solver='liblinear', penalty = 'l2')

model.fit(Xtrain, ytrain)

yFinal = model.predict(Xfinal) 

output =  pd.DataFrame({'PassengerId': initFinal.PassengerId, 'Survived':yFinal})

output.groupby('Survived').size()
model = tree.DecisionTreeClassifier(max_depth=3)

model.fit(Xtrain, ytrain)

yFinal = model.predict(Xfinal) 

output =  pd.DataFrame({'PassengerId': initFinal.PassengerId, 'Survived':yFinal})

output.groupby('Survived').size()



# Decision tree visualisation

tree.export_graphviz(model, './data/tree.dot', feature_names=columns)

# dot -Tpng data/tree.dot -o data/tree.png
output.to_csv('.titanic_{0}_{1}.csv'.format(type(model).__name__, accuracy), index=False)