import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid', palette='deep')

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

%matplotlib inline
df_raw = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_raw.info()
df_raw.head(5)
df_test.head(5)
df_test.shape
df_user= df_test['PassengerId']
df_user = pd.DataFrame(df_user, columns=['PassengerId'])
df_user.shape
df_raw.columns.values
new_columns = ['passangerId','survival', 'class', 'name', 'sex', 'age', 'siblings/spouses',

               'parents/children', 'ticket', 'fare', 'cabin', 'embarked']

new_columns_test = ['passangerId', 'class', 'name', 'sex', 'age', 'siblings/spouses',

               'parents/children', 'ticket', 'fare', 'cabin', 'embarked']
df = pd.DataFrame(df_raw.values, columns= new_columns )

df.info()
df_test = pd.DataFrame(df_test.values, columns= new_columns_test )

df_test.info()
df.head()
df['family'] = df['siblings/spouses'] + df['parents/children'] + 1

df = df.drop(['siblings/spouses','parents/children'], axis=1)

df['embarked'].value_counts()

df['embarked'].replace(['S', 'C', 'Q'], 

  ['southampton', 'cherbourg', 'quennstone'], inplace= True )
df_test['family'] = df_test['siblings/spouses'] + df_test['parents/children'] + 1

df_test = df_test.drop(['siblings/spouses','parents/children'], axis=1)

df_test['embarked'].value_counts()

df_test['embarked'].replace(['S', 'C', 'Q'], 

  ['southampton', 'cherbourg', 'quennstone'], inplace= True )
df[['class', 'survival', 'age', 'fare', 'passangerId',

    'family']] = df[['class',  'survival', 'age', 'fare','passangerId',

    'family']].apply(pd.to_numeric)



df_test[['class', 'age', 'fare', 'passangerId',

    'family']] = df_test[['class', 'age', 'fare','passangerId',

    'family']].apply(pd.to_numeric)
df.info()
df_test.info()
#Visualising Dataset

bins = range(0,100,10)



g = pd.crosstab(df.sex, df.survival).plot(kind='bar', figsize=(10,5))

ax = g.axes

for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points')  

plt.grid(b=True, which='major', linestyle='--')

plt.title('Survival Frequency for Genre')

plt.legend(['Not Survived', 'Survived'])

plt.xlabel('Genre')

plt.ylabel('Quantity')

plt.show()
g = df.groupby(pd.cut(df.age, bins))['age'].count().plot(kind='bar', figsize=(10,10))

ax = g.axes

for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

plt.grid(b=True, which='major', linestyle='--')

plt.title('Frequency of Age')

plt.grid(b=True, which='major', linestyle='--')

plt.xlabel('Age')

plt.ylabel('Quantity')

plt.show()
g = pd.crosstab(pd.cut(df.age, bins), df.survival).plot(kind='bar', figsize=(10,10))

ax = g.axes

for p in ax.patches:

     ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='gray', rotation=0, xytext=(0, 10),

         textcoords='offset points') 

plt.grid(b=True, which='major', linestyle='--')

plt.title('Survival Frequency for Age')

plt.legend(['Not Survived', 'Survived'])

plt.yticks(np.arange(0,250,50))

plt.xlabel('Age')

plt.ylabel('Quantity')

plt.show()
age_notsurvival = (df.groupby(pd.cut(df.age, bins))['age'].count()/ len(df[df.survival==0]))*100

age_survival = (df.groupby(pd.cut(df.age, bins))['age'].count()/ len(df[df.survival==1]))*100

age_notsurvival.shape

age_notsurvival.plot(kind='bar', figsize=(10,10))

plt.grid(b=True, which='major', linestyle='--')

plt.title('Percentage of Age for Passanger Not Survived')

plt.yticks(np.arange(0,110,10))

plt.xlabel('Age')

plt.ylabel('Percentage')

plt.show()
age_survival.plot(kind='bar', figsize=(10,10))

plt.grid(b=True, which='major', linestyle='--')

plt.title('Percentage of Age for Passanger Survived')

plt.yticks(np.arange(0,110,10))

plt.xlabel('Age')

plt.ylabel('Percentage')

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(10,10))

plt.subplots_adjust(hspace=0)

plt.suptitle('Age Frequency')

ax1 = sns.countplot(pd.cut(df.age, bins), data= df, 

                    color='darkblue', ax=axes[0], saturation=0.5)

ax2 = sns.countplot(pd.cut(df.age, bins)[df.survival==0], data=df , 

                    color='red', ax=axes[1], saturation=1, alpha=0.5)

ax2.set_xlabel('Age')

ax3 = sns.countplot(pd.cut(df.age, bins)[df.survival==1], data= df, 

                    color='darkblue', ax=ax2, saturation=1, alpha=0.5)

ax2.legend(['Have Not Survived', 'Have Survived'])
pd.crosstab(df['class'], df.survival).plot(kind='bar', figsize=(15,10))

plt.grid(b=True, which= 'major', linestyle='--')

plt.title('Survival Frequency for Class')

plt.yticks(np.arange(0,600,50))

plt.legend(['Not Survived', 'Survived'])

plt.xlabel('class')

plt.ylabel('Quantity')

plt.show()
pd.crosstab(df.embarked, df.survival).plot(kind='bar', figsize=(15,10))

plt.grid(b=True, which='major', linestyle='--')

plt.yticks(np.arange(0,700,50))

plt.title('Survival Frequency for Embarked')

plt.legend(['Not Survived', 'Survived'])

plt.xlabel('Embarked')

plt.ylabel('Quantity')

plt.show()
sns.pairplot(data=df, hue='survival', vars=['age', 'fare', ])
sns.countplot(x='survival', data=df)
sns.heatmap(data= df.corr(),annot=True,cmap='viridis')
sns.distplot(df.age, bins=10)
pd.crosstab(df.survival[df.embarked=='southampton'],df['class']).plot(kind='bar', figsize=(15,10))

plt.title('Survival Frequency for Class / Embarked(Southampton)')

plt.grid(b=True, which='Major', linestyle='--')

plt.legend(['First Class', 'Second Class', 'Third Class'])

plt.ylabel('Quatity')

plt.xlabel('Survival')

plt.show()
df.drop(['passangerId', 'survival'], axis=1).hist(figsize=(10,10))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
## Correlation with independent Variable (Note: Models like RF are not linear like these)

df2 = df.drop(['passangerId', 'survival'], axis=1)

df2.corrwith(df.survival).plot.bar(

        figsize = (10, 10), title = "Correlation with Survival", fontsize = 15,

        rot = 45, grid = True)
sns.set(style="white")

# Compute the correlation matrix

corr = df2.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
## Pie Plots (Just for binary values)

df.columns

df2 = df[['class','survival','sex', 'embarked']]

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, df2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(df2.columns.values[i - 1])

   

    values = df2.iloc[:, i - 1].value_counts(normalize = True).values

    index = df2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
df.describe()   
df.survival.value_counts()
countNotsurvival = len(df[df.survival == 0])     

countSurvival = len(df[df.survival == 1]) 

print('Percentage of Titanic not survived: {:.2f}%'.format((countNotsurvival/len(df)) * 100)) 

print('Percentage of Titanic survived: {:.2f}%'.format((countSurvival/len(df)) * 100))
df.groupby(df['survival']).mean()
#Looking for Null Values

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#Looking for Null Values

sns.heatmap(df_test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
df.isnull().any()
df_test.isnull().any()
df.isnull().sum()
df_test.isnull().sum()
null_percentage = (df.isnull().sum()/len(df) * 100)

null_percentage = pd.DataFrame(null_percentage, columns = ['Percentage Train Null Values (%)'])

null_percentage_test = (df_test.isnull().sum()/len(df_test) * 100)

null_percentage_test = pd.DataFrame(null_percentage_test, columns = ['Percentage Test Null Values (%)'])
print(null_percentage)

print(null_percentage_test)
#Define X and y

df.columns

X_train = df.drop(['passangerId', 'survival', 'name', 'ticket', 'cabin',

              'embarked'], axis=1)

y_train = df['survival']

df_test = df_test.drop(['passangerId',  'name', 'ticket', 'cabin',

              'embarked'], axis=1)
y_train.head()
#Get Dummies

X_train = pd.get_dummies(X_train)
df_test = pd.get_dummies(df_test)
X_train.head(5)
df_test.head()
#Avoiding Dummies Trap

X_train.columns

X_train = X_train.drop(['sex_male'], axis=1)

X_train.isnull().sum()
#Avoiding Dummies Trap

df_test.columns

df_test = df_test.drop(['sex_male'], axis=1)

df_test.isnull().sum()
#Taking care of Missing Values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(X_train.iloc[:, 1:2])

X_train.iloc[:, 1:2] = imputer.transform(X_train.iloc[:, 1:2])

X_train.isnull().sum()
#Taking care of Missing Values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

imputer = imputer.fit(df_test.iloc[:, 1:2])

df_test.iloc[:, 1:2] = imputer.transform(df_test.iloc[:, 1:2])

df_test.isnull().sum()
df_test[df_test['class'] == 3]['fare'].mean()
df_test[df_test.fare.isnull()] = df_test[df_test['class'] == 3]['fare'].mean()
df_test.isnull().sum()
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X_train.columns.values)

df_test = pd.DataFrame(sc_x.transform(df_test), columns=df_test.columns.values)
X_train.head()
df_test.head()
X_test = df_test
X_test.head()
## Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l1')

lr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = lr_classifier.predict(X_test)

acc = round(lr_classifier.score(X_train, y_train) * 100, 2)



results = pd.DataFrame([['Logistic Regression (Lasso)', acc]],

               columns = ['Model', 'Accuracy'])

acc
from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier(n_neighbors=25, metric='minkowski', p= 2)

kn_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = kn_classifier.predict(X_test)

acc = round(kn_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## SVM (Linear)

from sklearn.svm import SVC

svc_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

svc_linear_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svc_linear_classifier.predict(X_test)

acc = round(svc_linear_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['SVM (Linear)', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## SVM (rbf)

from sklearn.svm import SVC

svc_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

svc_rbf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svc_rbf_classifier.predict(X_test)

acc = round(svc_rbf_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['SVM (RBF)', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## Naive Bayes

from sklearn.naive_bayes import GaussianNB

gb_classifier = GaussianNB()

gb_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gb_classifier.predict(X_test)

acc = round(gb_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

dt_classifier.fit(X_train, y_train)



#Predicting the best set result

y_pred = dt_classifier.predict(X_test)

acc = round(dt_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Decision Tree', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'gini')

rf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = rf_classifier.predict(X_test)

acc = round(rf_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Random Forest gini (n=100)', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
## Ada Boosting

from sklearn.ensemble import AdaBoostClassifier

ad_classifier = AdaBoostClassifier()

ad_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = ad_classifier.predict(X_test)

acc = round(ad_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Ada Boosting', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
##Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gr_classifier = GradientBoostingClassifier()

gr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gr_classifier.predict(X_test)

acc = round(gr_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Gradient Boosting', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
##Xg Boosting

from xgboost import XGBClassifier

xg_classifier = XGBClassifier()

xg_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = xg_classifier.predict(X_test)

acc = round(xg_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Xg Boosting', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
##Ensemble Voting Classifier

from sklearn.ensemble import VotingClassifier

voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),

                                                  ('kn', kn_classifier),

                                                  ('svc_linear', svc_linear_classifier),

                                                  ('svc_rbf', svc_rbf_classifier),

                                                  ('gb', gb_classifier),

                                                  ('dt', dt_classifier),

                                                  ('rf', rf_classifier),

                                                  ('ad', ad_classifier),

                                                  ('gr', gr_classifier),

                                                  ('xg', xg_classifier),],

voting='soft')
for clf in (lr_classifier,kn_classifier,svc_linear_classifier,svc_rbf_classifier,

            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,

            voting_classifier):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, round(clf.score(X_train, y_train) * 100, 2))
# Predicting Voting Test Set

y_pred = voting_classifier.predict(X_test)

acc = round(voting_classifier.score(X_train, y_train) * 100, 2)



model_results = pd.DataFrame([['Ensemble Voting', acc]],

               columns = ['Model', 'Accuracy'])



results = results.append(model_results, ignore_index = True)

acc
results
#The Best Classifier

print('The best classifier is:')

print('{}'.format(results.sort_values(by='Accuracy',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=gr_classifier, X=X_train, y=y_train,cv=10)

accuracies.mean()

accuracies.std()

print("Gradient Boosting Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
X_test.shape
df_user.shape

y_pred.shape
submission = pd.DataFrame({

        "PassengerId": df_user["PassengerId"],

        "Survived": y_pred

    })
submission.head()
submission.to_csv('titanic_submission.csv', index=False)
print(submission)