#For data handling

import numpy as np

import pandas as pd



#For plotting

import matplotlib.pyplot as plt

import seaborn as sns



#Import raw data

raw_train = pd.read_csv('/kaggle/input/titanic/train.csv')

raw_test = pd.read_csv('/kaggle/input/titanic/test.csv')

raw_train.head()
raw_test.head()
print('Information about the train set:\n')

print(raw_train.info(), '\n'*3,'*'*50, '\n'*3)

print('Information about the test set:\n')

print(raw_test.info())
#Histogram of the 'Age' feature

plt.figure(figsize=(25,10))

sns.distplot(raw_train['Age'])
#Barplot of the 'Embarked' feature

plt.figure(figsize=(25,10))

sns.barplot(x=raw_train.groupby(by='Embarked').count().index, y=raw_train.groupby(by='Embarked').count().iloc[:,0])
print('Pclass:', raw_train['Pclass'].unique())

print('Name:', raw_train['Name'].unique().shape)

print('Sex:', raw_train['Sex'].unique())

print('Age:', raw_train['Age'].unique().shape)

print('SibSp:', raw_train['SibSp'].unique())

print('Parch:', raw_train['Parch'].unique())

print('Ticket:', raw_train['Ticket'].unique().shape)

print('Fare:', raw_train['Fare'].unique().shape)

print('Cabin:', raw_train['Cabin'].unique().shape)

print('Embarked:', raw_train['Embarked'].unique())
print('Pclass:', raw_test['Pclass'].unique())

print('Name:', raw_test['Name'].unique().shape)

print('Sex:', raw_test['Sex'].unique())

print('Age:', raw_test['Age'].unique().shape)

print('SibSp:', raw_test['SibSp'].unique())

print('Parch:', raw_test['Parch'].unique())

print('Ticket:', raw_test['Ticket'].unique().shape)

print('Fare:', raw_test['Fare'].unique().shape)

print('Cabin:', raw_test['Cabin'].unique().shape)

print('Embarked:', raw_test['Embarked'].unique())
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import train_test_split



#Features Selection

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'PassengerId']

train = raw_train[np.append(columns, ['Survived'])].copy().set_index('PassengerId')

test = raw_test[columns].copy().set_index('PassengerId')



#Override OneHotEncoder to have the column names created automatically (Title_Mr, Title_Mrs...) 

class OneHotEncoder(SklearnOneHotEncoder):

    def __init__(self, **kwargs):

        super(OneHotEncoder, self).__init__(**kwargs)

        self.fit_flag = False



    def fit(self, X, **kwargs):

        out = super().fit(X)

        self.fit_flag = True

        return out



    def transform(self, X, categories, index='', name='', **kwargs):

        sparse_matrix = super(OneHotEncoder, self).transform(X)

        new_columns = self.get_new_columns(X=X, name=name, categories=categories)

        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)

        return d_out



    def fit_transform(self, X, categories, index, name, **kwargs):

        self.fit(X)

        return self.transform(X, categories=categories, index=index, name=name)



    def get_new_columns(self, X, name, categories):

        new_columns = []

        for j in range(len(categories)):

            new_columns.append('{}_{}'.format(name, categories[j]))

        return new_columns



#Create a class for the transformation which will help use again the fitted encoder/scaler for the test set

class data_preparation():

    def __init__(self):

        self.enc_sex = LabelEncoder()

        self.scaler_age = StandardScaler()

        self.enc_title = LabelEncoder()

        self.onehotenc_title = OneHotEncoder(handle_unknown='ignore')

        self.onehotenc_pclass = OneHotEncoder(handle_unknown='ignore')

        self.enc_embarked = LabelEncoder()

        self.onehotenc_embarked = OneHotEncoder(handle_unknown='ignore')

        

        

    def transform(self, data_untouched):

        data = data_untouched.copy(deep=True)

        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

        

        #For the test set/Already fitted

        try:

            data['Sex'] = self.enc_sex.transform(data['Sex'].values)

            data['Age'] = self.scaler_age.transform(data['Age'].values.reshape(-1, 1))

            data['Title'] = data['Title'].map(lambda x: self.enc_title.transform([x])[0] if x in self.enc_title.classes_ else -1)

            data['Embarked'] = data['Embarked'].map(lambda x: self.enc_embarked.transform([str(x)])[0] if x in self.enc_embarked.classes_ else -1)

            data = pd.concat([self.onehotenc_title.transform(data['Title'].values.reshape(-1,1), categories=self.enc_title.classes_,name='Title', index=data.index),

                              self.onehotenc_pclass.transform(data['Pclass'].values.reshape(-1,1), categories=[3,2,1], name='Pclass', index=data.index),

                              self.onehotenc_embarked.transform(data['Embarked'].values.reshape(-1,1), categories=self.enc_embarked.classes_, name='Embarked', index=data.index),

                              data],axis=1)

        

        #For the train set/Not yet fitted    

        except NotFittedError:

            self.mean_age = data['Age'].median()

            self.median_fare = data['Fare'].median()

            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)

            data['Age'] = self.scaler_age.fit_transform(data['Age'].values.reshape(-1, 1))

            data['Title'] = self.enc_title.fit_transform(data['Title'].values)

            data['Embarked'] = self.enc_embarked.fit_transform(data['Embarked'].astype(str).values)

            data = pd.concat([self.onehotenc_title.fit_transform(data['Title'].values.reshape(-1,1), categories=self.enc_title.classes_, name='Title', index=data.index), 

                              self.onehotenc_pclass.fit_transform(data['Pclass'].values.reshape(-1,1), categories=[3,2,1], name='Pclass', index=data.index),

                              self.onehotenc_embarked.fit_transform(data['Embarked'].values.reshape(-1,1), categories=self.enc_embarked.classes_, name='Embarked', index=data.index),

                              data], axis=1)



        

        data['Family_size'] = data['SibSp'] + data['Parch']

        data['IsAlone'] = (data['Family_size'] == 0).astype(int)

        data['Age'] = data['Age'].fillna(self.mean_age)

        data['Fare'] = data['Fare'].fillna(self.median_fare)

        data = data.drop(columns=['Pclass', 'Embarked', 'Title', 'Name'])

            

        return data



#Apply the transformation

data_prep = data_preparation()

X_train_full = data_prep.transform(train.drop(columns='Survived'))

y_train_full = train['Survived']

X_test = data_prep.transform(test)



#Fix Randomness and split the train and validation set

np.random.seed(0)

X_train, X_val, y_train, y_val =  train_test_split(X_train_full, y_train_full, test_size=0.10, random_state=50)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.1, size=15)

sns.heatmap(X_train_full[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']].astype(float).corr(method='pearson'),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



#Model

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.ensemble import VotingClassifier
#Dictionnary of parameters to try

list_params = {

    'Tree':{'max_depth': np.logspace(2,8,6)},

    'SVC':{'C' : np.linspace(0.01,1,3), 'kernel':['poly'], 'degree':np.linspace(2,8,4).astype(int)},

    'KNN':{'n_neighbors' : np.linspace(2,30,29).astype(int)},

    'gauss':{'var_smoothing' : np.logspace(-8,-10,3)},

    'bern':{'alpha' : np.linspace(0.1,1,10)}

}



#Base models

list_base_model = {

    'Tree' : DecisionTreeClassifier(random_state=0),

    'SVC' : SVC(random_state=0),

    'KNN' : KNeighborsClassifier(),

    'gauss' : GaussianNB(),

    'bern' : BernoulliNB()

}



#Apply GridSearch, fit and 

list_models = pd.DataFrame(list_params.items(), columns = ['name', 'params']).set_index('name')

list_models['model'] = list_base_model.values()

list_models['grid_model'] = list_models.apply(lambda x: GridSearchCV(x['model'], x['params']), axis=1)

list_models['grid_model'] = list_models.apply(lambda x: x['grid_model'].fit(X_train, y_train), axis=1)

list_models['score'] = list_models.apply(lambda x: x['grid_model'].predict(X_val), axis=1, result_type='reduce').apply(lambda x: accuracy_score(x, y_val))

print('Score for each base model using GridSearch:')

print(list_models['score'])
models = (('Tree', DecisionTreeClassifier(**list_models.loc['Tree', 'grid_model'].best_params_, random_state=40)),

          #We have to enable the probability estimates for the soft voting.

          ('SVC', SVC(**list_models.loc['SVC', 'grid_model'].best_params_, random_state=40, probability=True)),

          ('KNN', KNeighborsClassifier(**list_models.loc['KNN', 'grid_model'].best_params_)),

          ('gauss', GaussianNB(**list_models.loc['gauss', 'grid_model'].best_params_)),

          ('bern', BernoulliNB(**list_models.loc['bern', 'grid_model'].best_params_)))



vot = VotingClassifier(models, voting='hard')

vot.fit(X_train,y_train)

print('Score of the VotingClassifer with hard voting:', accuracy_score(vot.predict(X_val), y_val))
vot = VotingClassifier(models, voting='soft')

vot.fit(X_train,y_train)

print('Score of the VotingClassifer with soft voting:', accuracy_score(vot.predict(X_val), y_val))
vot = VotingClassifier(models, voting='soft', weights=list_models['score']/list_models['score'].sum())

vot.fit(X_train,y_train)

print('Score of the VotingClassifer with soft weighted voting:', accuracy_score(vot.predict(X_val), y_val))
vot = VotingClassifier(models, voting='soft')

vot.fit(X_train_full,y_train_full)

submission = pd.DataFrame({'PassengerId': test.index, 'Survived': vot.predict(X_test)})

submission.to_csv('submission_stack.csv', index=False)
from sklearn.ensemble import StackingClassifier

from sklearn.neural_network import MLPClassifier



models = (('Tree', DecisionTreeClassifier(**list_models.loc['Tree', 'grid_model'].best_params_, random_state=40)),

                   ('SVC', SVC(**list_models.loc['SVC', 'grid_model'].best_params_, random_state=40, probability=True)),

                   ('KNN', KNeighborsClassifier(**list_models.loc['KNN', 'grid_model'].best_params_)),

                   ('gauss', GaussianNB(**list_models.loc['gauss', 'grid_model'].best_params_)),

                   ('bern', BernoulliNB(**list_models.loc['bern', 'grid_model'].best_params_)))



final_estimator = MLPClassifier(hidden_layer_sizes = (100,50,), random_state=1)

stack = StackingClassifier(models, final_estimator=final_estimator)

stack.fit(X_train,y_train)

print('Score of the StackingClassifer with hard voting:', accuracy_score(stack.predict(X_val), y_val))
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



#Base models

models = (('Tree', DecisionTreeClassifier(**list_models.loc['Tree', 'grid_model'].best_params_, random_state=40)),

                   ('SVC', SVC(**list_models.loc['SVC', 'grid_model'].best_params_, random_state=40, probability=True)),

                   ('KNN', KNeighborsClassifier(**list_models.loc['KNN', 'grid_model'].best_params_)),

                   ('gauss', GaussianNB(**list_models.loc['gauss', 'grid_model'].best_params_)),

                   ('bern', BernoulliNB(**list_models.loc['bern', 'grid_model'].best_params_)))



#First layer of stacking (MLP + Logistic Regression)

MLP = MLPClassifier(hidden_layer_sizes = (50,5,), random_state=1)

stack_MLP = StackingClassifier(models, final_estimator=MLP)



log = LogisticRegression()

stack_log = StackingClassifier(models, final_estimator=log)



QDA = QuadraticDiscriminantAnalysis()

stack_QDA = StackingClassifier(models, final_estimator=QDA)



#Second layer of stack (small MLP)

small_MLP = MLPClassifier(hidden_layer_sizes = (5,), random_state=10)

stack_final = StackingClassifier([('MLP',stack_MLP), ('log', stack_log), ('QDA', stack_QDA)], final_estimator=small_MLP)

stack_final.fit(X_train,y_train)

print('Score of the StackingClassifer:', accuracy_score(stack_final.predict(X_val), y_val))
from sklearn.ensemble import BaggingClassifier



#Bagging models

list_base_model = {

    'Tree' : BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=100, max_samples=0.5, random_state=10),

    'SVC' : BaggingClassifier(SVC(C=0.2), n_estimators=20, max_samples=0.5, random_state=10),

    'KNN' : BaggingClassifier(KNeighborsClassifier(n_neighbors=8, weights='distance'), n_estimators=50, max_samples=0.3, random_state=10),

    'gauss' : BaggingClassifier(GaussianNB(), n_estimators=100, max_samples=0.5, random_state=10),

    'bern' : BaggingClassifier(BernoulliNB(alpha=0.01, binarize=0.5), n_estimators=200, max_samples=0.4, random_state=10)

}



list_models_bag = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_bag['model'] = list_models_bag.apply(lambda x: x['model'].fit(X_train, y_train), axis=1)

list_models_bag['score'] = list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='reduce').apply(lambda x: accuracy_score(x, y_val))

print('Score for Bagging model on samples:')

print(list_models_bag['score'])
print('Accuracy using all model:',accuracy_score(list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='expand').mode().values.reshape(-1,), y_val))
list_base_model = {

    'Tree' : BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=10, max_features=0.7, random_state=10),

    'SVC' : BaggingClassifier(SVC(C=0.2), n_estimators=50, max_features=0.3, random_state=10),

    'KNN' : BaggingClassifier(KNeighborsClassifier(n_neighbors=8, weights='distance'), n_estimators=10, max_features=0.7, random_state=10),

    'gauss' : BaggingClassifier(GaussianNB(), n_estimators=10, max_features=0.7, random_state=10),

    'bern' : BaggingClassifier(BernoulliNB(alpha=0.01, binarize=0.5), n_estimators=10, max_features=0.7, random_state=10)

}



list_models_bag = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_bag['model'] = list_models_bag.apply(lambda x: x['model'].fit(X_train, y_train), axis=1)

list_models_bag['score'] = list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='reduce').apply(lambda x: accuracy_score(x, y_val))

print('Score for Bagging model on features:')

print(list_models_bag['score'])
print('Accuracy using all model:',accuracy_score(list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='expand').mode().values.reshape(-1,), y_val))
list_base_model = {

    'Tree' : BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=10, max_features=0.7, max_samples=0.5, random_state=10),

    'SVC' : BaggingClassifier(SVC(C=0.2), n_estimators=50, max_features=0.4, max_samples=0.4, random_state=10),

    'KNN' : BaggingClassifier(KNeighborsClassifier(n_neighbors=8, weights='distance'), n_estimators=50, max_features=0.5, max_samples=0.5, random_state=10),

    'gauss' : BaggingClassifier(GaussianNB(), n_estimators=50, max_features=0.5, max_samples=0.5, random_state=10),

    'bern' : BaggingClassifier(BernoulliNB(alpha=0.01, binarize=0.5), n_estimators=50, max_features=0.5, max_samples=0.5, random_state=10)

}



list_models_bag = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_bag['model'] = list_models_bag.apply(lambda x: x['model'].fit(X_train, y_train), axis=1)

list_models_bag['score'] = list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='reduce').apply(lambda x: accuracy_score(x, y_val))

print('Score for Bagging model on features:')

print(list_models_bag['score'])
print('Accuracy using all model:',accuracy_score(list_models_bag.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='expand').mode().values.reshape(-1,), y_val))
#Train again on the whole train set and predict on the test set

list_models_bag = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_bag['model'] = list_models_bag.apply(lambda x: x['model'].fit(X_train_full, y_train_full), axis=1)

my_submission = pd.DataFrame({'PassengerId': test.index, 'Survived': list_models_bag.apply(lambda x: x['model'].predict(X_test), axis=1, result_type='expand').mode().values.reshape(-1,)})

my_submission.to_csv('submission_bag.csv', index=False)
from sklearn.ensemble import AdaBoostClassifier



list_base_model = {

    'Tree' : AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=20, random_state=0, learning_rate=1.05),

    'SVC' : AdaBoostClassifier(SVC(C=0.3), algorithm='SAMME',n_estimators=5, random_state=0, learning_rate=1.1),

    'gauss' : AdaBoostClassifier(GaussianNB(), n_estimators=5, random_state=0, learning_rate=1.3),

    'bern' : AdaBoostClassifier(BernoulliNB(alpha=0.01, binarize=0.5), n_estimators=5, random_state=0, learning_rate=1.3)

}



list_models_boost = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_boost['model'] = list_models_boost.apply(lambda x: x['model'].fit(X_train, y_train), axis=1)

list_models_boost['score'] = list_models_boost.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='reduce').apply(lambda x: accuracy_score(x, y_val))

print('Score for Boosting model:')

print(list_models_boost['score'])
print('Accuracy using all model:',accuracy_score(list_models_boost.apply(lambda x: x['model'].predict(X_val), axis=1, result_type='expand').mode().iloc[0,:].values.reshape(-1,), y_val))
#Train again on the whole train set and predict on the test set

list_models_boost = pd.DataFrame(list_base_model.items(), columns = ['name', 'model']).set_index('name')

list_models_boost['model'] = list_models_boost.apply(lambda x: x['model'].fit(X_train_full, y_train_full), axis=1)

my_submission = pd.DataFrame({'PassengerId': test.index, 'Survived': list_models_boost.apply(lambda x: x['model'].predict(X_test), axis=1, result_type='expand').mode().iloc[0,:].astype(int).values.reshape(-1,)})

my_submission.to_csv('submission_boost.csv', index=False)