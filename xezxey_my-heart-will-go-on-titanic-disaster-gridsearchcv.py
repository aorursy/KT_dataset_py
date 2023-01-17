import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data sets
titanic_train_dataset = pd.read_csv("../input/train.csv")
titanic_test_dataset = pd.read_csv("../input/test.csv")
titanic_submission_form_dataset = pd.read_csv("../input/gender_submission.csv")

#Preview the data
titanic_train_dataset.head(n=5)
#By running .info method you will see some missing value in ['Age'] and ['Cabin'] Columns than we will clean this in the next part
titanic_train_dataset.info()

#Filling some missing data
#Age
mean_age = titanic_train_dataset['Age'].mean()
std_age = titanic_train_dataset['Age'].std()
titanic_train_dataset['Age'] = titanic_train_dataset['Age'].fillna(np.random.randint(low = mean_age - std_age, high = mean_age + std_age))
titanic_train_dataset[['Age']].describe()    #Re-Check that we have fill all of missing data
#print(titanic_train_dataset['Name'].loc[titanic_train_dataset['Age'] < 1])
titanic_train_dataset['CategoricalAge'] = pd.qcut(titanic_train_dataset['Age'], q = 4)

#CabinFloorScore
#Use regular expression to match floor pattern
#Set default floor for 3rd class ticket(GuestRoom in floor F and G)
import re
cabin_pattern = re.compile("[a-zA-Z]")
GuestRoom_random_floor = ["F", "G"]

cabin_floor_list = []
for cabin in titanic_train_dataset['Cabin']:
    if pd.isnull(cabin):
        cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
    else:
        cabin_floor = re.findall(cabin_pattern, cabin)
        cabin_floor_list.append(min(cabin_floor))

titanic_train_dataset['Cabin'] = cabin_floor_list

titanic_train_dataset.info()
titanic_train_dataset.head(5)

#FamilySize
titanic_train_dataset['FamilySize'] = titanic_train_dataset['SibSp'] + titanic_train_dataset['Parch'] + 1

#IsAlone
titanic_train_dataset.loc[titanic_train_dataset['FamilySize'] > 1, 'IsAlone'] = 0
titanic_train_dataset.loc[titanic_train_dataset['FamilySize'] == 1, 'IsAlone'] = 1

#FarePerPerson
titanic_train_dataset['FarePerPerson'] = titanic_train_dataset['Fare'] / titanic_train_dataset['FamilySize']
titanic_train_dataset['CategoricalFarePerPerson'] = pd.qcut(titanic_train_dataset['FarePerPerson'], q = 4)

#CabinFloorScore
#Use regular expression to match floor pattern
#Set default floor for 3rd class ticket(GuestRoom in floor F and G)
import re
cabin_pattern = re.compile("[a-zA-Z]")
GuestRoom_random_floor = ["F", "G"]

cabin_floor_list = []
for cabin in titanic_train_dataset['Cabin']:
    if pd.isnull(cabin):
        cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
    else:
        cabin_floor = re.findall(cabin_pattern, cabin)
        cabin_floor_list.append(min(cabin_floor))

titanic_train_dataset['CabinFloor'] = cabin_floor_list

#Giving a score for each cabin floor.
titanic_train_dataset["CabinFloorScore"] = 0
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "T"] = 7 
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "A"] = 6
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "B"] = 5
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "C"] = 4
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "D"] = 3
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "E"] = 2
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "F"] = 1
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "G"] = 0

#TitleScore
name_title = []
title_pattern = re.compile("[a-zA-Z]{1,}\.")
for name in titanic_train_dataset['Name']:
    name_title.append(re.findall(title_pattern, name)[0])
    
titanic_train_dataset['Title'] = name_title
#print(set(titanic_train_dataset['Title']))
                                                                                            
#We will give each title a score 
#1.Have power : Don., Col., Major., Capt., Sir., Mme., Lady., Countess.    => Score = 3
#2.Adult : Mr., Mrs., Miss., MS., Mlle., Dr., Rev.,                        => Score = 2
#3.Child : Jonkheer, Master                                                => Socre = 1

prior1_title_influence = ['Don.', 'Col.', 'Major.', 'Capt.', 'Sir.', 'Mme.', 'Lady.', 'Countess.']
prior2_title_adult = ['Mr.', 'Miss.', 'Ms.', 'Mlle.', 'Mrs.', 'Dr.', 'Rev.']
prior3_title_kid = ['Jonkheer.', 'Master.']

titanic_train_dataset["TitleScore"] = 0
titleScore_list = []

def give_title_score(title):
    if title in prior1_title_influence:
        return 3
    elif title in prior2_title_adult: 
        return 2
    elif title in prior3_title_kid: 
        return 1
    else: 
        return 0

for i in range(len(titanic_train_dataset)):
    titleScore_list.append(give_title_score(titanic_train_dataset['Title'][i]))

titanic_train_dataset['TitleScore'] = titleScore_list

#Try to use the linear regression to predict missing age
plt.figure(9)
plt.scatter(titanic_train_dataset['Pclass'], titanic_train_dataset['Age'], c='red')
plt.xlabel('Pclass')
plt.ylabel('Age')

plt.figure(10)
plt.scatter(titanic_train_dataset['FarePerPerson'], titanic_train_dataset['Age'], c='blue')
plt.xlabel('FarePerPerson')
plt.ylabel('Age')

plt.figure(11)
plt.scatter(titanic_train_dataset['Sex'], titanic_train_dataset['Age'], c='green')
plt.xlabel('Sex')
plt.ylabel('Age')

plt.figure(12)
plt.scatter(titanic_train_dataset['IsAlone'], titanic_train_dataset['Age'], c='yellow')
plt.xlabel('IsAlone')
plt.ylabel('Age')

plt.figure(13)
plt.scatter(titanic_train_dataset['FamilySize'], titanic_train_dataset['Age'], c='black')
plt.xlabel('FamilySize')
plt.ylabel('Age')
#There're no correlation between [Age] feature and other features. So I will use second approach(Random between [mean-std, mean+std]).

plt.rcParams["figure.figsize"] = (10, 7)
#1. Sex Vs. Survived
plt.figure(1)
sex_vs_survived = titanic_train_dataset[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
print(sex_vs_survived)
plt.bar(sex_vs_survived['Sex'], sex_vs_survived['Survived'], tick_label=sex_vs_survived['Sex'], width = 0.5)
plt.title('Sex Vs. Survived')
plt.xlabel('Sex')
plt.ylabel('Survived Rate')

#2. Age Vs. Survived
plt.figure(2)
age_vs_survived = titanic_train_dataset[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
age_vs_survived['AgeGroup'] = ['Child to Youth', 'Youth to Middle Aged', 'Middle Aged' ,'Middle Aged to Old']
print(age_vs_survived)
plt.bar(age_vs_survived['AgeGroup'], age_vs_survived['Survived'], width=0.5)
plt.title('Age Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Age')

#3. Pclass Vs. Survived
plt.figure(3)
titanic_train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
pclass_vs_survived = titanic_train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
print(pclass_vs_survived)
plt.bar(pclass_vs_survived['Pclass'], pclass_vs_survived['Survived'], tick_label=pclass_vs_survived['Pclass'])
plt.title('Pclass Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Pclass')

#4. FamilySize Vs. Survived
plt.figure(4)
famsize_vs_survived = titanic_train_dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
print(famsize_vs_survived)
plt.bar(famsize_vs_survived['FamilySize'], famsize_vs_survived['Survived'], tick_label=famsize_vs_survived['FamilySize'])
plt.title('FamilySize Vs. Survived')
plt.xlabel('FamilySize')
plt.ylabel('Survived Rate')
plt.plot(famsize_vs_survived['FamilySize'], famsize_vs_survived['Survived'], color='Red')

#5. IsAlone Vs. Survived
plt.figure(5)
isalone_vs_survived = titanic_train_dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
print(isalone_vs_survived)
plt.bar(isalone_vs_survived['IsAlone'], isalone_vs_survived['Survived'], tick_label=isalone_vs_survived['IsAlone'])
plt.title('IsAlone Vs. Survived')
plt.xlabel('IsAlone')
plt.ylabel('Survived Rate')

#6. CabinFloorScore Vs. Survived
plt.figure(6)
cabinfloorscore_vs_survived = titanic_train_dataset[['CabinFloorScore', 'Survived']].groupby(['CabinFloorScore'], as_index=False).mean()
print(cabinfloorscore_vs_survived)
plt.bar(cabinfloorscore_vs_survived['CabinFloorScore'], cabinfloorscore_vs_survived['Survived'], tick_label=cabinfloorscore_vs_survived['CabinFloorScore'])
plt.title('CabinFloorScore Vs. Survived')
plt.xlabel('CabinFloorScore')
plt.ylabel('Survived Rate')

#7. FarePerPerson Vs. Survived
plt.figure(7)
fareperperson_vs_survived = titanic_train_dataset[['CategoricalFarePerPerson', 'Survived']].groupby(['CategoricalFarePerPerson'], as_index=False).mean()
fareperperson_vs_survived['TicketGrade'] = ['Very Cheap', 'Cheap', 'Moderate', 'Expensive']
print(fareperperson_vs_survived)
plt.bar(fareperperson_vs_survived['TicketGrade'], fareperperson_vs_survived['Survived'])
plt.title('FarePerPerson Vs. Survived')
plt.xlabel('FarePerPerson')
plt.ylabel('Survived Rate')

#8. TitleScore Vs. Survived
plt.figure(8)
titlescore_vs_survived = titanic_train_dataset[['TitleScore', 'Survived']].groupby(['TitleScore'], as_index=False).mean()
print(titlescore_vs_survived)
plt.bar(titlescore_vs_survived['TitleScore'], titlescore_vs_survived['Survived'], tick_label=['Influence', 'Adult', 'Kids'])
plt.title('TitleScore Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Title')
#Preparing dataset for training and testing 
#Creating the function names "create_feature" for create features from given data.
def create_feature(df, mode):
    """
    create_feature function 
    1. Input parameters : 
        1.1 df : Dataframe variable for input dataset that you want to create feature from it.
        1.2 mode : Mode selection (1 is for training dataframe, 0 for testing dataframe)
            Different between 2 mode is the selected columns
            - Training dataframe mode will return all of features include 'Survived' Columns
            - Testing dataframe mode will return only feature
    2. Returned value :
        2.1 df : return a dataframe after finish a creating feature step
    3. Function process : Take df and mode as input. Then create features and slice columns that we need to use for training and testing (depend on mode)
    """
    
    #Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['CategoryFamilySize'] = pd.cut(df['FamilySize'], bins=5)
    
    #IsAlone
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    #CabinFloor
    #Split cabin out of room number
    import re
    cabin_pattern = re.compile("[a-zA-Z]")
    GuestRoom_random_floor = ["F", "G"]
    
    cabin_floor_list = []
    #print(titanic_train_dataset["Name"].loc[titanic_train_dataset['Cabin'] > "O"])
    for cabin in df['Cabin']:
        if pd.isnull(cabin):
            cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
        else:
            cabin_floor = re.findall(cabin_pattern, cabin)
            cabin_floor_list.append(min(cabin_floor))
    
    df['CabinFloor'] = cabin_floor_list
    
    #Score for each cabin floor
    df["CabinFloorScore"] = 0
    df["CabinFloorScore"].loc[df['CabinFloor'] == "T"] = 7
    df["CabinFloorScore"].loc[df['CabinFloor'] == "A"] = 6
    df["CabinFloorScore"].loc[df['CabinFloor'] == "B"] = 5
    df["CabinFloorScore"].loc[df['CabinFloor'] == "C"] = 4
    df["CabinFloorScore"].loc[df['CabinFloor'] == "D"] = 3
    df["CabinFloorScore"].loc[df['CabinFloor'] == "E"] = 2
    df["CabinFloorScore"].loc[df['CabinFloor'] == "F"] = 1
    df["CabinFloorScore"].loc[df['CabinFloor'] == "G"] = 0
    
    #Find how many passenger in each floor
    cabin_floor_list = ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    #for i in range(len(cabin_floor_list)):
    #    print("Cabin " + cabin_floor_list[i] + " : " + str(df["CabinFloorScore"].loc[df['CabinFloor'] == cabin_floor_list[i]].count()))
    
    #In test set : There's missing [fare] value as NaN. So I will fill this with mean
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))   
    
    #Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['CategoricalFarePerPerson'] = pd.qcut(df['FarePerPerson'], q = 4)
    

    
    
    #VIP
    #Impact
    name_title = []
    title_pattern = re.compile("[a-zA-Z]{1,}\.")
    for name in df['Name']:
        name_title.append(re.findall(title_pattern, name)[0])
        
    df['Title'] = name_title
                                                                                                
    #We will give each title a score 
    #1.Have power : Don., Col., Major., Capt., Sir., Mme., Lady., Countess.    => Score = 3
    #2.Adult : Mr., Mrs., Miss., MS., Mlle., Dr., Rev.,                        => Score = 2
    #3.Child : Jonkheer, Master                                                => Socre = 1
    
    prior1_title_powerful = ['Don.', 'Col.', 'Major.', 'Capt.', 'Sir.', 'Mme.', 'Lady.', 'Countess.']
    prior2_title_adult = ['Mr.', 'Miss.', 'Ms.', 'Mlle.', 'Mrs.', 'Dr.', 'Rev.']
    prior3_title_kid = ['Jonkheer.', 'Master.']
    
    df["TitleScore"] = 0
    TitleScore_list = []
    
    def give_title_score(title):
        if title in prior1_title_powerful:
            return 3
        elif title in prior2_title_adult:
            return 2
        elif title in prior3_title_kid : 
            return 1
        else: 
            return 0
    
    for i in range(len(df.index)):
        TitleScore_list.append(give_title_score(df['Title'][i]))
    
    df['TitleScore'] = TitleScore_list    
    
    #Age
    #No correlation between each features and age   
    #Age have some missing values ---> Use mean, median or std to fill the nan
    #Random between [mean-std, mean+std]
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(low = mean_age - std_age, high = mean_age + std_age))
    df[['Age']].describe()    #Re-Check that we have fill all of missing data
    df['CategoricalAge'] = pd.qcut(titanic_train_dataset['Age'], q = 4)
    
    
    if mode:
        df = df.loc[:, ["Survived", "Pclass", "Sex", "Age", "FamilySize", 
                                                              "IsAlone", "CabinFloorScore", "FarePerPerson", 
                                                              "TitleScore"]]
    else:
        df = df.loc[:, ["Pclass", "Sex", "Age", "FamilySize", 
                                                              "IsAlone", "CabinFloorScore", "FarePerPerson", 
                                                              "TitleScore"]]
    
    #Encoding string into number : Male = 1, Female = 0
    from sklearn.preprocessing import LabelEncoder
    labelencoder_sex = LabelEncoder()
    df['Sex'] = labelencoder_sex.fit_transform(df['Sex'].values)
    return df

#Calling function
titanic_train_dataset = pd.read_csv("../input/train.csv")    #Re-import a titanic data set
titanic_train_dataset_for_training_step = create_feature(df=titanic_train_dataset, mode=1)
titanic_test_dataset_for_testing_step = create_feature(df=titanic_test_dataset, mode=0)

#Showing some data
titanic_train_dataset_for_training_step.head(3)
titanic_test_dataset_for_testing_step.head(3)

#To ensure your data is ready to train and test 
titanic_train_dataset_for_training_step.info()
titanic_test_dataset_for_testing_step.info()

X_train = titanic_train_dataset_for_training_step.iloc[:, 1:]
y_train = titanic_train_dataset_for_training_step.iloc[:, 0]
#Now, It's ready for training

#Training Phase
#1.Model Usage : D.Tree, Forest, ANNs, Logistic Regression, KNN, Naive Bayes
#2.Training Strategy : Grid Search
#3.Testing Strategy : K-fold Cross Validation

#Step 1. Creating model object from each class
#Importing Model libraries
from sklearn.neighbors import KNeighborsClassifier    #KNN
from sklearn.tree import DecisionTreeClassifier    #D.Tree
from sklearn.ensemble import RandomForestClassifier #Forest
from sklearn.naive_bayes import GaussianNB    #Naive Bayes
from sklearn.linear_model import LogisticRegression    #Logistic Regression

#Importing Model Validation
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.cross_validation import cross_val_score

#Initial Model Object from Class
clf_knns = KNeighborsClassifier()
clf_dtree = DecisionTreeClassifier()
clf_forest = RandomForestClassifier()
clf_logreg = LogisticRegression()
clf_naive = GaussianNB()

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    ]

classifiers = [clf_knns, clf_dtree, clf_forest, clf_logreg]
    
#Step 2. Delcare parameters for training and do a hyperparameter tuning by gridsearch
# You can change your parameters here!!!
params_knns = [{'n_neighbors' : range(1, 100)}, {'metric' : ['minkowski']}, {'p' : [2]}]
params_dtree = [{'criterion' : ['gini', 'entropy']}, {'splitter' : ['random', 'best']}]
params_forest = [{'n_estimators' : range(1, 100)}, {'criterion':['entropy', 'gini']}]
params_logreg = [{'penalty' : ['l1', 'l2']}, {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}, {'max_iter' : range(100, 1000)}]

parameters = [
        params_knns,
        params_dtree,
        params_forest,
        params_logreg,
        ]

#Step 2. Training model using gridsearch
#Importing Grid Search
from sklearn.model_selection import GridSearchCV

clf_best_acc = []
clf_best_params = []
clf_best_estimator = []

        
grid_searchs = [] #grid_search_knns, grid_search_dtree, grid_search_forest, grid_search_svm, grid_search_logreg, grid_search_naive

#Training and append all of the best result from each model to a list
for i in range(len(classifiers)):
    grid_searchs.append(GridSearchCV(estimator=classifiers[i], param_grid=parameters[i], scoring='accuracy', cv=10, 
                                n_jobs=-1))   
    grid_searchs[i].fit(X_train, y_train)

    clf_best_acc.append(grid_searchs[i].best_score_)
    clf_best_params.append(grid_searchs[i].best_params_)
    clf_best_estimator.append(grid_searchs[i].best_estimator_)
 
print("Finishing Training")
#best_classifier variable for storing best classifier from gridsearch as a dict : Key is the name of classifier, Value is list of [best accuracy, best parameters, best estimators]
best_classifier = {}

#Store each classifier in dictionary 
for i in range(len(classifiers)):
    best_classifier[classifiers[i].__class__.__name__] = [clf_best_acc[i], clf_best_params[i], clf_best_estimator[i]]

#Print out the result of each best classifier can do!!!
for key, value in best_classifier.items():
    print("Classifier name : " + str(key), end="\n")
    print("Accuracy : " + str(value[0]), end="\n")    #value[0] is best acccuracy
    print("Best parameters : " + str(value[1]), end="\n")    #value[1] is best parameters
    print("Best estimator : " + str(value[2]), end="\n")    #value[2] is best estimators
    print("************************************************************************************************", end="\n")
   
#Comparing between each model performance
import os
import numpy as np
import matplotlib.pyplot as plt

x = best_classifier.keys()
y = list(value[0]*100 for key, value in best_classifier.items())

fig, ax = plt.subplots()   

width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color=['blue', 'red', 'green', 'purple'])    # Make bar plot in horizontal line
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(y):
    ax.text(v+.7, i , str(v) + '%', color='red', fontweight='bold')
plt.title('Model Performance')
plt.xlabel('Classifier Performance(Percentage)')
plt.ylabel('Classifier name')   

#Testing on test set and create a submission file for submitting to kaggle.
y_pred_submission = pd.DataFrame(grid_searchs[2].predict(titanic_test_dataset_for_testing_step))
y_pred_submission['PassengerId'] = titanic_test_dataset['PassengerId']
y_pred_submission.columns = ['Survived', 'PassengerId']
y_pred_submission = y_pred_submission[['PassengerId', 'Survived']]

#Make sure we have the same format for submission to kaggle
y_pred_submission.head(3)
titanic_submission_form_dataset.head(3)

y_pred_submission.info()
titanic_submission_form_dataset.info()

#Let's goooooooooooooo
y_pred_submission.to_csv('forest_submission.csv', index=False)

