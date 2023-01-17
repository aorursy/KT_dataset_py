##### Importing Libraries #####

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib

from matplotlib import pyplot as plt



##### Setting Options #######

matplotlib.style.use('ggplot')

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100
###### Reading Train and Test Csv #####

data = pd.read_csv('../input/train.csv')
##### Imputing Missing Values for Age ######

data['Age'].fillna(data['Age'].median(), inplace=True)

data.describe()
###### Gender Vs Survival Chart ######

survived_sex = data[data['Survived']==1]['Sex'].value_counts()

dead_sex = data[data['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_sex,dead_sex])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(9,6))
##### Embarkment Vs Survival 

survived_embark = data[data['Survived']==1]['Embarked'].value_counts()

dead_embark = data[data['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar',stacked=True, figsize=(9,6))
####### Age Vs Survival Chart ####

figure = plt.figure(figsize=(9,6))

plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
######## Fare Vs Survival #####

figure = plt.figure(figsize=(9,6))

plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],

         bins = 30,label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
##### Age Vs Fare Vs Survival ####

plt.figure(figsize=(9,6))

ax = plt.subplot()

ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)

ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
########## Combining Test And Train Data For Feature Engineering #####

def get_combined_data():

    # reading train data

    train = pd.read_csv('../input/train.csv')

    

    # reading test data

    test = pd.read_csv('../input/test.csv')



    # extracting and then removing the targets from the training data 

    targets = train.Survived

    train.drop('Survived',1,inplace=True)

    



    # merging train data and test data for future feature engineering

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop('index',inplace=True,axis=1)

    

    return combined



combined = get_combined_data()

combined.shape
####### Extracting the passenger titles #####

def get_titles():



    global combined

    

    # we extract the title from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"



                        }

    

    # we map each title

    combined['Title'] = combined.Title.map(Title_Dictionary)

    

get_titles()
##### Filling Missing Values In Age Column #####

combined["Age"] = combined.groupby(['Sex','Pclass','Title'])['Age'].transform(lambda x: x.fillna(x.median()))
def process_names():

    

    global combined

    # we clean the Name variable

    combined.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    

    # removing the title variable

    combined.drop('Title',axis=1,inplace=True)

    

process_names()
##### Processing Fare ######

def process_fares():

    

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.Fare.mean(),inplace=True)

    

process_fares()
###### Processing Embarked #####



def process_embarked():

    

    global combined

    # two missing embarked values - filling them with the most frequent one (S)

    combined.Embarked.fillna('S',inplace=True)

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)



process_embarked()
##### Processing Cabin #####

def process_cabin():

    

    global combined

    

    # replacing missing cabins with U (for Unknown)

    combined.Cabin.fillna('U',inplace=True)

    

    # mapping each Cabin value with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])

    

    # dummy encoding ...

    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')

    

    combined = pd.concat([combined,cabin_dummies],axis=1)

    

    combined.drop('Cabin',axis=1,inplace=True)



process_cabin()
##### Processing Gender #####

def process_gender():

    

    global combined

    # mapping string values to numerical one 

    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})

    

process_gender()
#### Processing Pclass



def process_pclass():

    

    global combined

    # encoding into 3 categories:

    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")

    

    # adding dummy variables

    combined = pd.concat([combined,pclass_dummies],axis=1)

    

    # removing "Pclass"

    

    combined.drop('Pclass',axis=1,inplace=True)

    



process_pclass()
def process_ticket():

    

    global combined

    

    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

    def cleanTicket(ticket):

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        ticket = map(lambda t : t.strip() , ticket)

        ticket = list(filter(lambda t : not t.isdigit(), ticket))

        if len(ticket) > 0:

            return ticket[0]

        else: 

            return 'XXX'

    



    # Extracting dummy variables from tickets:



    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')

    combined = pd.concat([combined, tickets_dummies],axis=1)

    combined.drop('Ticket',inplace=True,axis=1)



ticket  = process_ticket()
####### Processing Family ######



def process_family():

    

    global combined

    # introducing a new feature : the size of families (including the passenger)

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    # introducing other features based on the family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

    

process_family()
###### Split Test And Train #####

def recover_train_test_target():

    global combined

    

    train0 = pd.read_csv('../input/train.csv')

    

    targets = train0.Survived

    train = combined.ix[0:890]

    test = combined.ix[891:]

    

    return train,test,targets



train,test,targets = recover_train_test_target()
from sklearn import linear_model

from sklearn import metrics

from sklearn.metrics import roc_curve, auc,confusion_matrix,classification_report
# Initialize logistic regression model

log_model = linear_model.LogisticRegression()



# Train the model

log_model.fit(X = train,y = targets)



# Make predictions

preds = log_model.predict(X= train)



# Check trained model intercept

print (log_model.intercept_)



# Check trained model coefficients

print (log_model.coef_)


def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_true=targets,y_pred=preds)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=["Dead","Survived"],

                      title='Confusion matrix')



# Plot normalized confusion matrix

# plt.figure()

# plot_confusion_matrix(cnf_matrix, classes=["Dead","Survived"], normalize=True,

#                       title='Normalized confusion matrix')



plt.show()
###### Constructing ROC And AUC ####

fpr, tpr, threshold = metrics.roc_curve(targets, preds)

roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('1 - False Positive Rate')

plt.show()
# Make test set predictions

test_preds = log_model.predict(X=test)



# Create a submission for Kaggle

submission = pd.DataFrame({"PassengerId":test["PassengerId"],

                           "Survived":test_preds})



# Save submission to CPassengerId,Survived

submission.to_csv("tutorial_logreg_submission.csv", index=False)       
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=200)

clf = clf.fit(train, targets)

features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_
features.sort(['importance'],ascending=False)
model = SelectFromModel(clf, prefit=True)

train_new = model.transform(train)

print (train_new.shape)





test_new = model.transform(test)

print (test_new.shape)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV



forest = RandomForestClassifier(max_features='sqrt')



parameter_grid = {

                 'max_depth' : [4,5,6,7,8],

                 'n_estimators': [200,210,240,250],

                 'criterion': ['gini','entropy']

                 }



cross_validation = StratifiedKFold(targets, n_folds=5)



grid_search = GridSearchCV(forest,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(train_new, targets)



preds = grid_search.predict(train_new).astype(int)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_true=targets,y_pred=preds)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=["Dead","Survived"],title='Confusion matrix')



# Plot normalized confusion matrix

# plt.figure()

# plot_confusion_matrix(cnf_matrix, classes=["Dead","Survived"], normalize=True,

#                       title='Normalized confusion matrix')



plt.show()
###### Constructing ROC And AUC ####

fpr, tpr, threshold = metrics.roc_curve(targets, preds)

roc_auc = metrics.auc(fpr, tpr)

plt.title('ROC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
# Make test set predictions

test_preds = grid_search.predict(X=test_new)



# Create a submission for Kaggle

submission = pd.DataFrame({"PassengerId":test["PassengerId"],"Survived":test_preds})



# Save submission to CSV

submission.to_csv("tutorial_random_forest_submission.csv", index=False)