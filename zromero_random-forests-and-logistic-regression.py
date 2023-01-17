#Basic Data analysis and munging (if needed)

import pandas as pd

import numpy as np



#sklearn and all the things I used.

from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression



#Seaborn/matplotlib to visualize stuff.

import seaborn as sns

from matplotlib import pyplot as plt

#Changing these settings allows us to view the entirety of the collumns avoiding that "..." in the middle of the df prints.

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



#Data Set being used is the Mushroom Data set from UCI ML

#Found here https://www.kaggle.com/uciml/mushroom-classification



df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")



df.head()
df.info()

le = LabelEncoder()



for column in df:

    df[column] = le.fit_transform(df[column])

    

df.info()


df.head()



df.describe()
# I Create an X variable here purely because it's easier for me to recall what I was doing when I go back and look at this.

# When it comes to classification in datasets when we split this dataset having duplicates will likely lead to an 

# Unreasonably high accuracy.

# We can create our X and remove any duplicates if there were any.

X = df.drop_duplicates(keep='first')



#the classification "feature" and our final Dataset to Split

y = X['class']

X = X.drop(['class'], axis=1)

#I used a random state to test the random forest classifier parameters.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
# n_estimators is the number of trees we will use

# Max_depth determins the amount of features to be used in classification

# Once again using a random state to keep the same sets, I could alter criterion as well.

# The random Forest

forest_trials = []

tree_count = [x*5 for x in range(21) if x*5 >= 5]

for n in tree_count:

    for depth in range(1,10):

        forest = RandomForestClassifier(n_estimators=n, max_depth=depth , random_state=50, criterion='gini')  

        forest.fit(X_train, y_train)

        forest_trials.append({'Depth':depth,'Trees':n, 'cfm':confusion_matrix(y_test, forest.predict(X_test))})
#Just Checking to see if I was appending the data in a way I can work with.

print(forest_trials[0])
#Creat X/Y to plot to show the accuracy using True Positive + True Negative / total predictions.



tp_fp = {'Accuracy':[],'Trees':[], 'Depth':[]}



for trial in forest_trials:

    true_negative = trial['cfm'][0][0]

    false_positive = trial['cfm'][0][1]

    false_negative = trial['cfm'][1][0]

    true_positive = trial['cfm'][1][1]

    summed = true_positive+true_negative+false_positive+false_negative

    tp_fp['Trees'].append(trial['Trees'])

    tp_fp['Depth'].append(trial['Depth']) 

    tp_fp['Accuracy'].append(round(((true_positive+true_negative)/summed),2)*100)





tp_fp['Accuracy'] = tuple(tp_fp['Accuracy'])

tp_fp['Trees']= tuple(tp_fp['Trees'])

tp_fp['Depth'] = tuple(tp_fp['Depth'])

sns.set(style="darkgrid")



ax = sns.lineplot(x='Trees', y="Accuracy", hue="Depth", legend='brief', data=tp_fp)

ax.set(xlabel='Number of Trees', ylabel='Accuracy', title="Tree Count vs Depth")

ax.legend(ncol=3,fontsize= 'large', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title="Tree Depth", labels="123456789")



plt.show()
sns.set(style="darkgrid")





plt.show()

sns.set(style="ticks")



ax = sns.scatterplot(x="Trees", y="Depth", hue='Accuracy', palette='muted', legend="brief", data=tp_fp)

ax.set(xlabel='Number of Trees', ylabel='Depth', title="Tree VS Depth Prediction Improvements")

ax.legend(ncol=3,fontsize= 'large',bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,title="Correct Predictions (%)")



finalVals= []

solvers = {"newton-cg":["l2",'none'], "lbfgs":["l2", 'none'], "liblinear":["l2","l1"], "sag":["l2", 'none'], "saga":["l2","l1", 'none']}

for solver in solvers:

    for penalty in range(len(solvers[solver])):

        model = LogisticRegression(max_iter = 50000, solver=solver,penalty=solvers[solver][penalty])

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        cfm = confusion_matrix(y_test, prediction)

        true_negative = cfm[0][0]

        false_positive = cfm[0][1]

        false_negative = cfm[1][0]

        true_positive = cfm[1][1]

        total_correct= round(((true_negative+true_positive) / (true_negative+true_positive+false_positive+false_negative)),2)

        finalVals.append([solver, solvers[solver][penalty], total_correct])

  

graph = {"solver":[],"penalty":[],"accuracy":[]}

for each_pred in finalVals:

    graph["solver"].append(each_pred[0])

    graph["penalty"].append(each_pred[1])

    graph["accuracy"].append(each_pred[2])

    

ax = sns.scatterplot(x="solver", y="accuracy", hue='penalty', palette='muted', legend="brief", data=graph)

ax.set(xlabel='Solver', ylabel='Accuracy')

ax.legend(ncol=1,fontsize= 'large',bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, title="Regularization Type (Penalty)")