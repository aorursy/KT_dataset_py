#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
#Read the training & test data
liver_df = pd.read_csv('../input/indian_liver_patient.csv')
liver_df.head()
liver_df.info()
#Describe gives statistical information about NUMERICAL columns in the dataset
liver_df.describe(include='all')
#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.
#Gender has only 2 values - Male/Female
#Which features are available in the dataset?
liver_df.columns
#Check for any null values
liver_df.isnull().sum()
sns.countplot(data=liver_df, x = 'Dataset', label='Count')

LD, NLD = liver_df['Dataset'].value_counts()
print('Percentage of patients diagnosed with liver disease: ',LD / (LD+NLD) * 100)

sns.countplot(data=liver_df, x = 'Gender', label='Count')

M, F = liver_df['Gender'].value_counts()
malesWithLiverDisease = liver_df[(liver_df['Gender'] == 'Male') & (liver_df['Dataset'] == 1)]['Age'].count()
femalesWithLiverDisease = liver_df[(liver_df['Gender'] == 'Female') & (liver_df['Dataset'] == 1)]['Age'].count()
patientsWithLiverDisease = liver_df[liver_df['Dataset'] == 1]['Age'].count()
totalPatients = liver_df['Age'].count()
print('Percent of patients that have liver disease: ',patientsWithLiverDisease /totalPatients * 100)
print('Percent of male patients that have liver disease: ',malesWithLiverDisease /M * 100)
print('Percent of female patients that have liver disease: ',femalesWithLiverDisease /F * 100)

print('Number of patients that are male: ',M)
print('Number of patients that are female: ',F)
liver_df[['Dataset','Age']].groupby(['Dataset']).mean()
#liverDisease_df = liver_df[liver_df['Dataset'] == 1]
#liverDisease_df.drop(['Gender', 'Dataset'], axis=1).boxplot()
#nonLiverDisease_df = liver_df[liver_df['Dataset'] == 2]
#nonLiverDisease_df.drop(['Gender', 'Dataset'], axis=1).boxplot()
fig=plt.figure(figsize=(20, 24), dpi= 80, facecolor='w', edgecolor='k')

ax = liver_df.drop(['Gender'], axis='columns').set_index('Dataset', append=True).stack().to_frame().reset_index().rename(columns={'level_2':'quantity', 0:'value'}).pipe((sns.boxplot,'data'), x='quantity', y='value', hue='Dataset')
ax.set(ylim=(0,500))
# Correlation
liver_corr = liver_df.corr()
liver_corr
plt.figure(figsize=(15, 15))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 10},
           cmap= 'coolwarm')
plt.title('Correlation between features');
liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].median())

X = liver_df.drop(['Gender', 'Dataset'], axis='columns')
X.head()
X.describe()
y = liver_df['Dataset'] # 1 for liver disease; 2 for no liver disease
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score

def processResults(clk, str="", X_test = X_test, X_train = X_train, y_test = y_test, y_train = y_train):
    predicted = clk.predict(X_test)
    score = round(clk.score(X_train, y_train) * 100, 2)
    score_test = round(clk.score(X_test, y_test) * 100, 2)

    print(str + 'Training score: \n', score)
    print(str + 'Test Score: \n', score_test)
    print('Accuracy: \n', accuracy_score(y_test,predicted))
    print(confusion_matrix(y_test,predicted))
    print(classification_report(y_test,predicted))
    sns.heatmap(confusion_matrix(y_test,predicted),annot=True,fmt="d")
    return score, score_test
    
from sklearn.linear_model import LogisticRegression

#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)

logreg_score, logreg_score_test = processResults(logreg, "Logistic Regression ")

print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gauss_score, gauss_score_test = processResults(gaussian, "Gaussian Naive Bayesian ")
# Decision tree
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_score, decision_tree_score_test  = processResults(decision_tree, "Decision Tree ")

# As max depth is 14, let's reduce it, until the test and train data scores are close to each other
def plotDecisionTreeScoresVSDepth(clk, maxDepth = 14, X_test = X_test, X_train = X_train, y_test = y_test, y_train = y_train):
    score = []
    score_test = []
    allDepth = np.arange(maxDepth,1,-1)
    for depth in allDepth:
        clk.set_params(**{'random_state': 42, 'max_depth' : depth})
        clk.fit(X_train, y_train)
        
        predicted = clk.predict(X_test)
        score.append(round(clk.score(X_train, y_train) * 100, 2))
        score_test.append(round(clk.score(X_test, y_test) * 100, 2))
    plt.plot(allDepth, score)    
    plt.plot(allDepth, score_test)
    plt.ylabel('Accuracy')
    plt.xlabel('Max depth of decision tree')
    plt.legend(['Train accuracy', 'Test accuracy'])
    plt.show()

decision_tree = DecisionTreeClassifier()        
plotDecisionTreeScoresVSDepth(decision_tree)
# Random Forest
# The random forest classifier uses number of decision trees and combines the results to make a prediction
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

random_forest_score, random_forest_score_test  = processResults(random_forest, "Random Forest ")
randomForest = RandomForestClassifier(n_estimators=100)        
plotDecisionTreeScoresVSDepth(randomForest)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),    
    ('pca', PCA(n_components=8)),
    ('svc', SVC()),
])
pipe.fit(X_train, y_train)
svcScore, svcScore_test = processResults(pipe)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scale', StandardScaler()),    
    ('pca', PCA(n_components=8)),
    ('svc', SVC()),
])
pipe.set_params(svc__C=2)
pipe.fit(X_train, y_train)
svcScore, svcScore_test = processResults(pipe)

from sklearn.neighbors import KNeighborsClassifier

pipeknn = Pipeline([
    ('scale', StandardScaler()),    
    ('knn', KNeighborsClassifier(n_neighbors=5)),
])
pipeknn.fit(X_train, y_train)
knnTrainScore, knnTestScore = processResults(pipeknn)
## K nearest neighbors

liver_df.head()
###Model evaluation
#We can now rank our evaluation of all the models to choose the best one for our problem. 
models = pd.DataFrame({
    'Model': [ 'Logistic Regression', 'Gaussian Naive Bayes','Decision Tree', 'Random Forest', 'Support Vector Classifier', 'Nearest Neighbour'],
    'Score': [ logreg_score, gauss_score, decision_tree_score, random_forest_score, svcScore, knnTrainScore],
    'Test Score': [ logreg_score_test, gauss_score_test, decision_tree_score_test, random_forest_score_test, svcScore_test, knnTestScore]})
models.sort_values(by='Test Score', ascending=False)