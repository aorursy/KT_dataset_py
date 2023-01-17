import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from imblearn.combine import SMOTETomek
df = pd.read_csv('../input/performance-prediction/summary.csv')
df.head()
df.info()
df.isnull().sum()
df = df.fillna(0)
columns = df.columns.tolist()[1:]
plt.figure(figsize=(20,20))
sns.heatmap(df[columns].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.title('Correlations Heat Map')
plt.show()
bins = np.arange(10,df.GamesPlayed.max(),5)
plt.figure(figsize=(10,7))
plt.hist(df[df.Target == 1].GamesPlayed,alpha=0.8,bins=bins)
plt.hist(df[df.Target == 0].GamesPlayed,alpha=0.8,bins=bins)
plt.title('Difference in Games played by the Target')
plt.xlabel('Number of Games Played')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.show()
df['efficiency'] = (df['FieldGoalsMade']+df['Rebounds']+df['Assists']+df['Steals']+df['Blocks']+df['Turnovers'])/df['MinutesPlayed']
df['Participation'] = df['MinutesPlayed']/df['GamesPlayed']
bins = np.arange(0.2,df.Participation.max(),0.1)
plt.figure(figsize=(10,7))
plt.hist(df[df.Target == 1]['efficiency'],alpha=0.8)
plt.hist(df[df.Target == 0]['efficiency'],alpha=0.8)
plt.title('Difference in Efficiency by the Target')
plt.xlabel('Efficiency Score')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.show()
bins = np.arange(0.2,df.Participation.max(),0.1)
plt.figure(figsize=(10,7))
plt.hist(df[df.Target == 1]['Participation'],alpha=0.8)
plt.hist(df[df.Target == 0]['Participation'],alpha=0.8)
plt.title('Difference in Participation by the Target')
plt.xlabel('Participation Score')
plt.ylabel('Frequency')
plt.xticks(bins)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0,1))
#Target
Y = df['Target'].values
#Inputs
X = df[['GamesPlayed', 'MinutesPlayed', 'PointsPerGame',
       'FieldGoalsMade', 'FieldGoalsAttempt', 'FieldGoalPercent', '3PointMade',
       '3PointAttempt', '3PointPercent', 'FreeThrowMade', 'FreeThrowAttempt',
       'FreeThrowPercent', 'OffensiveRebounds', 'DefensiveRebounds',
       'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
       'efficiency', ]].values
#Normalize our variables
X = mms.fit_transform(X)
#Split to training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
#Define the model
model = XGBClassifier(learning_rate = 0.1,n_estimators=200, max_depth=6)
#train the model
model.fit(X_train, y_train)
#Check training accuracy
trainingAccuracy =  metrics.accuracy_score(y_train,model.predict(X_train))
print("Training Accuracy: %.2f%%" % (trainingAccuracy * 100.0))
#Check testing accuracy
testingAccuracy =  metrics.accuracy_score(y_test, model.predict(X_test))
print("Testing Accuracy: %.2f%%" % (testingAccuracy * 100.0))
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')
# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")