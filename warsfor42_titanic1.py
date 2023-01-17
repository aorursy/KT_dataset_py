import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score,roc_curve, auc,precision_recall_curve,accuracy_score
from sklearn.metrics import average_precision_score,classification_report,confusion_matrix
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt 
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

y_true = pd.read_csv("../input/genderclassmodel.csv")

# FAMILY
# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
train['Family'] =  train["Parch"] + train["SibSp"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0


# SEX
# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)
test['Person']    = test[['Age','Sex']].apply(get_person,axis=1)

train['Person'] = train[['Age','Sex']].apply(get_person,axis=1)
test['Person']    = test[['Age','Sex']].apply(get_person,axis=1)



#train['Person'] = train['Person'].map( {'female': 0, 'child': 1} ).astype(int)
#test['Person']    = train['Person'].map( {'female': 0, 'child': 1} ).astype(int)

"""
# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Male','Female','Child']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Male','Female','Child']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_titanic)
test  = test.join(person_dummies_test)
"""

# EMBARKED
# Embarked from 'C', 'Q', 'S'
if len(train.Embarked[ train.Embarked.isnull() ]) > 0:
    train.Embarked[ train.Embarked.isnull() ] = train.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train.Embarked = train.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

if len(test.Embarked[ test.Embarked.isnull() ]) > 0:
    test.Embarked[ test.Embarked.isnull() ] = test.Embarked.dropna().mode().values
test.Embarked = test.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# DROP SOME FEATURES
# Collect the test data's PassengerIds before dropping it
ids = train['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train = train.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','SibSp','Parch'], axis=1)
# Collect the test data's PassengerIds before dropping it
ids = test['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test = test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','SibSp','Parch'], axis=1) 


# AGE
# All the ages with no data -> make the median of all Ages
median_age = train['Age'].dropna().median()
if len(train.Age[ train.Age.isnull() ]) > 0:
    train.loc[ (train.Age.isnull()), 'Age'] = median_age
# All the ages with no data -> make the median of all Ages
median_age = test['Age'].dropna().median()
if len(test.Age[ test.Age.isnull() ]) > 0:
    test.loc[ (test.Age.isnull()), 'Age'] = median_age
    
# FARE
# All the missing Fares -> assume median of their respective class
if len(test.Fare[ test.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    # loop 0 to 2
    for f in range(0,3):                                              
        median_fare[f] = test[ test.Pclass == f+1 ]['Fare'].dropna().median()
    # loop 0 to 2
    for f in range(0,3):                                             
        test.loc[ (test.Fare.isnull()) & (test.Pclass == f+1 ), 'Fare'] = median_fare[f]
        

train= train.dropna()     
test = test.dropna()        

train['Person'] = train['Person'].map( {'female': 0, 'male':1,'child': 2} ).astype(int)
test['Person'] = train['Person'].map( {'female': 0, 'male':1,'child': 2} ).astype(int)
        
     
        
       
train

parameters = {'n_estimators':[50,10000,5000,2000],
              'max_depth' : [2,4,6],
              'min_samples_split':[2,4,6,20,30,25]
              }      
forest = RandomForestClassifier()

gs = GridSearchCV(forest, parameters, n_jobs=-1)
gs = gs.fit(train_data[0::,1::], train_data[0::,0])
best_parameters, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))
# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train.values
test_data = test.values


print ('Training...')
forest = RandomForestClassifier(max_depth= 5,min_samples_split= 2,n_estimators= 50)
forest = forest.fit(train_data[0::,1::], train_data[0::,0] )

print ('Predicting...')
output = forest.predict(test_data).astype(int)
print("Accuracy :",accuracy_score(y_true.Survived, output))
print("Rapport de classification:")
print(classification_report(y_true.Survived, output))
print("Matrice de confusion:")
def plot_confusion_matrix(cm, title='Matrice de confusion', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['No','Yes'], rotation=45)
    plt.yticks(tick_marks, ['No','Yes'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Compute confusion matrix
cm = confusion_matrix(y_true.Survived, output)
np.set_printoptions(precision=2)
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.figure()
plt.title("Validation curve")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(n_estimators=100),
                                                        train_data[0::,1::], train_data[0::,0],
                                                        train_sizes=np.linspace(.1, 1.0, 5), cv=5)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
# Determine the false positive and true positive rates
y_score = forest.predict_proba(test_data)
fpr, tpr, _ = roc_curve(y_true.Survived, y_score[:, 1])

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Compute Precision-Recall and plot curve
precision, recall, _ = precision_recall_curve(y_true.Survived, y_score[:, 1])
average_precision = average_precision_score(y_true.Survived, y_score[:, 1])
# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(np.array(y_true).ravel(),
                                                                np.array(y_score[:, 1]).ravel())
average_precision["micro"] = average_precision_score(y_true.Survived, y_score[:, 1],average="micro")
# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
plt.legend(loc="lower left")
plt.show()