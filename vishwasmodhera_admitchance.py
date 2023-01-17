#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

%matplotlib inline
sns.set_style("darkgrid")
#reading the csv file
data = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
#data description
data.describe()
#checking the first 5 rows
data.head()
#making the serial number the index
data = data.set_index("Serial No.")
#checking the last few lines
data.tail()
#checking the shape of the dataframe and the columns
print(data.shape)
print()
print(data.columns)
#renaming the columns
data = data.rename(columns = {'Serial No.' : 'Sno', 'GRE Score':'GRE', 'TOEFL Score':'TOEFL', 'University Rating':'UniRating', 'Chance of Admit':'Chance'})
#checking the first 5 rows again
data.head()
#checking the null values
data.isnull().sum()
#checking the information of the dataframe once again
data.info()
#we need to change the dtype of research as category
data['Research'] = data['Research'].astype('category')
data = data.rename(columns={'Chance of Admit ' : 'Chance'})
#checking the dtypes now
data.info()
sns.distplot(data['GRE'])
plt.title("Distribution of the GRE scores.")
sns.distplot(data['TOEFL'])
plt.title("Distribution of the TOEFL scores.")
sns.distplot(data['CGPA'])
plt.title("Distribution of the CGPAs.")
sns.catplot(x='Research', data=data, kind='count')
plt.title("Number of students who have done some research.")
#Now we are interested in relationships between different variables
f, ax = plt.subplots(figsize=(5,5), dpi=200)
ax = sns.kdeplot(data.CGPA, data.GRE, cmap="Reds", shade=True)
plt.title("CGPA vs GRE Scores")
f, ax = plt.subplots(figsize=(5,5), dpi=200)
ax = sns.kdeplot(data.CGPA, data.TOEFL, cmap="Blues", shade=True)
plt.title("CGPA vs TOEFL Scores")
data = data.rename(columns={'LOR ':'LOR'})
ax = sns.jointplot(data.SOP, data.LOR,  kind='reg')
data.columns
data.head()
x = data.loc[:,:]
f, ax = plt.subplots(figsize=(10,10), dpi = 300)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(x.corr(), annot=True, fmt='.3g', center=0, linewidths=0, cbar=True, square=True)
plt.title("Correlation of the variables with themselves")
#importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = data.drop(['Chance'], axis=1)
y = data.Chance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#lets do linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Linear Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Linear Regression R^2 score : ", model.score(X_test, y_test))

x = np.arange(1,101)
f, ax = plt.subplots(figsize=(5,5), dpi=200)
sns.scatterplot(x=x, y=y_test, color='r', alpha=0.3)
sns.regplot(x=x, y=prediction, color='b')
plt.title("Linear Regression")
#now we take a look at Ridge Regression
from sklearn.linear_model import Ridge

maxa = 0
maxs = 0
mina = 0
mins = 1


l = np.arange(0,10,0.01)
scorel = []
for i in l:
    model = Ridge(alpha=i)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scorel.append(score)
    if score > maxs:
        maxs = score
        maxa = i
    if score < mins:
        mins = score
        mina = i

model = Ridge(alpha=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Alpha of the Ridge : ", model.alpha)
print("Ridge Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Ridge Regression R^2 score : ", model.score(X_test, y_test))
predAlpha1 = prediction
print("-"*70)

model = Ridge(alpha=mina)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Alpha of the Ridge (worst alpha) : ", model.alpha)
print("Ridge Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Ridge Regression R^2 score : ", model.score(X_test, y_test))
predAlphaWorst = prediction
print("-"*70)

model = Ridge(alpha=maxa)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Alpha of the Ridge (best alpha) : ", model.alpha)
print("Ridge Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Ridge Regression R^2 score : ", model.score(X_test, y_test))
predAlphaBest = prediction
f, ax = plt.subplots(figsize=(20,5), dpi = 300)
plt.plot(predAlpha1, label='Prediction for Alpha = Unity', marker='o')
plt.plot(predAlphaBest, label='Prediction for Alpha = Best', marker='_' )
plt.plot(predAlphaWorst, label='Prediction for Alpha = Worst', marker='.')
plt.legend()
plt.title("Ridge Regreesion")
plt.show()
f, ax = plt.subplots(figsize=(15,10), dpi=300)
sns.scatterplot(x=l, y=scorel, marker='+')
plt.title("Alpha vs R^2 score for Ridge Regression")
f, ax = plt.subplots(figsize=(15,10), dpi=300)
sns.scatterplot(x=x, y=y_test, color='r', alpha=0.3)
sns.regplot(x=x, y=prediction, color='b')
plt.title("Ridge Regression")
#now we do lasso regression
from sklearn.linear_model import Lasso
x = data

maxa = 0
maxs = 0
mina = 0
mins = 1


l = np.arange(0.1,10,0.01)
scorel = []
for i in l:
    model = Lasso(alpha=i)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scorel.append(score)
    if score > maxs:
        maxs = score
        maxa = i
    if score < mins:
        mins = score
        mina = i

model = Lasso(alpha=1)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Coefficients of the Lasso model : ", model.coef_)
print("Lasso Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Lasso Regression R^2 score : ", model.score(X_test, y_test))
predAlpha1 = prediction
print("-"*70)

model = Lasso(alpha=mina)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Coefficients of the Lasso model (worst score) : ", model.coef_)
print("Lasso Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Lasso Regression R^2 score : ", model.score(X_test, y_test))
predAlphaWorst = prediction
print("-"*70)

model = Lasso(alpha=maxa)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("Coefficients of the Lasso model (best score) : ", model.coef_)
print("Lasso Regression mean squared error : ", np.sqrt(mean_squared_error(y_test, prediction)))
print("Lasso Regression R^2 score : ", model.score(X_test, y_test))
predAlphaBest = prediction

f, ax = plt.subplots(figsize=(20,5), dpi = 300)
plt.plot(predAlpha1, label='Prediction for Default Coefficients', marker='o')
plt.plot(predAlphaBest, label='Prediction for Best Coefficients', marker='_' )
plt.plot(predAlphaWorst, label='Prediction for Worst Coefficients', marker='.')
plt.legend()
plt.title("Lasso Regression")
plt.show()
f, ax = plt.subplots(figsize=(15,10), dpi=300)
sns.scatterplot(x=l, y=scorel, marker='+')
plt.title("Alpha vs R^2 score for Lasso Regression")
data.describe()
#Now we do cross validation for linear regression 
from sklearn.model_selection import cross_val_score

model = LinearRegression()
X = data.loc[:,:'CGPA']
y = data['Chance']
k = 5

cv_res = cross_val_score(model, X, y, cv=k)
print("Cross-validation scores : ", cv_res)
print("Cross-validation score average : ", np.sum(cv_res)/k)
#now we make a column in the dataframe that has categorical data
#the categorical data will be chance of getting an admit
data.describe()
#list comprehension for the new column
data['ChanceCat'] = ['Very likely' if i > 0.82 else 'Likely' if i > 0.72 else 'Somewhat likely' if i > 0.63 else 'Not likely' for i in data['Chance']]
#checking if the changes took place
data[['Chance','ChanceCat']].head()
#changing the data type of the newly created column as categorical data
data['ChanceCat'] = data['ChanceCat'].astype('category')
#plotting the ChanceCat
f, ax = plt.subplots(figsize=(8,6), dpi=200)
sns.boxenplot(x='ChanceCat', y='Chance', data=data, order=['Not likely', 'Somewhat likely', 'Likely', 'Very likely'])
plt.ylabel("Probability")
plt.title("Chance Distribution of getting admitted")
#Let's do logistic regression on the newly created categorical data
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
data['ChanceInt'] = [1 if i=='Likely' else 0 if i=='Not likely' else 0 if i=='Somewhat likely' else 1 for i in data['ChanceCat']]
data['ChanceInt'] = data['ChanceInt'].astype('category')
X = data.loc[:,'GRE':'Research']
X = np.array(X)
X = preprocessing.scale(X)
y = data['ChanceInt']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

f, ax = plt.subplots(figsize=(7,7), dpi=250)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
print("R^2 score : ", model.score(X_test, y_test))
X = data.loc[:,:'Chance']
y = data['ChanceCat']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prediction = model.predict_proba(X_test)
y_test_dummies = pd.get_dummies(y_test,drop_first=False).values

fpr = dict()
tpr = dict()
thresholds = dict()

for i in range(4):
    fpr[i],tpr[i],thresholds[i] = roc_curve(y_test_dummies[:,i],y_prediction[:,i])

f, ax = plt.subplots(figsize=(10,7), dpi=200)    
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr[0],tpr[0])
plt.plot(fpr[1],tpr[1])
plt.plot(fpr[2],tpr[2])
plt.plot(fpr[3],tpr[3])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.show()
print("R^2 score : ", model.score(X_test, y_test))
#Now let's take a look at KNN
data.columns
#the columns ChanceCat and ChanceInt are redundant values
#we should drop the column ChanceInt to save memory
data.drop('ChanceInt', axis=1)
#knn
from sklearn.neighbors import KNeighborsClassifier
X = data.loc[:,'GRE':'Research']
y = data['ChanceCat']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state=1)

train_score = []
test_score = []
k = np.arange(1,50)
for i in range(1,50):
    knn =KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    train_score.append(knn.score(X_train, y_train))
    test_score.append(knn.score(X_test, y_test))
    
f, ax = plt.subplots(figsize=(10,10), dpi=250)
sns.lineplot(x=k, y=train_score, label='Train Scores')
sns.lineplot(x=k, y=test_score, label ='Test Scores')
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Score")
plt.title("Number of neighbours vs Train Test scores")

print("Best accuracy is {} with k = {}".format(np.max(test_score), 1+test_score.index(np.max(test_score))))
#appling cross-validation in KNN

from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(X_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
f,ax = plt.subplots(figsize=(5,5), dpi=200)
sns.stripplot(x='ChanceCat', y='Chance', data=data, dodge=True, alpha=0.25, zorder=1, order=['Not likely', 'Somewhat likely', 'Likely', 'Very likely'])
sns.pointplot(x='ChanceCat', y='Chance', data=data, dodge=.523, join=False, palette='dark', markers='d', scale=.75, ci=None, order=['Not likely', 'Somewhat likely', 'Likely', 'Very likely'])
plt.title("Chances of getting admitted")

data.head()
l = np.arange(1,501)
f, ax = plt.subplots(figsize=(10,10), dpi=300)
sns.scatterplot(l,'Chance', data=data, hue='ChanceCat', palette=sns.diverging_palette(145, 280, s=85, l=25, n=4))
plt.title("Distribution of the Chances")
plt.xlabel("Index")
plt.ylabel("Probability")
#color palette = sns.diverging_palette(145, 280, s=85, l=25, n=7)
#sns.cubehelix_palette(4, start=2, rot=0, dark=0, light=.85, reverse=True)
#Generating feature importance
from sklearn.ensemble import RandomForestRegressor

classifier = RandomForestRegressor()
X = data.loc[:,'GRE':'Research']
y = data.loc[:,'Chance']

classifier.fit(X,y)
feature_names = X.columns
impFrame = pd.DataFrame()
impFrame['Features'] = X.columns
impFrame['Importance'] = classifier.feature_importances_
impFrame = impFrame.sort_values(by=['Importance'], ascending=True)
f, ax = plt.subplots(figsize=(10,10), dpi=200)
plt.barh([1,2,3,4,5,6,7], impFrame['Importance'], align='center', alpha=0.5)
plt.yticks([1,2,3,4,5,6,7], impFrame['Features'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()
X = data.loc[:,'GRE':'Chance']
sns.pairplot(X, hue='Chance', palette=sns.dark_palette("purple"))
X = data.loc[:,:'ChanceCat']
sns.pairplot(X, hue='ChanceCat', hue_order=['Very likely','Likely','Somewhat likely','Not likely'], palette=sns.diverging_palette(145, 280, s=85, l=25, n=8))
print(impFrame)