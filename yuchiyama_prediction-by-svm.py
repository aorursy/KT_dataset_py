import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
class BaseTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
class DataFrameDuplicator(BaseTransformer):
    
    def transform(self, df, y=None):
        return df.copy()
import re

class TitleExtractor(BaseTransformer):

    def __init__(self):
        self.__pattern = re.compile(r'[A-Z][^ ]*')
        self.__title_map = {
            'Mr'       : 'Mr',
            'Miss'     : 'Miss',
            'Mrs'      : 'Mrs',
            'Master'   : 'Master',
            'Capt'     : 'Mr',
            'Col'      : 'Mr',
            'Countess' : 'Mrs',
            'Don'      : 'Mr',
            'Dona'     : 'Mrs',
            'Dr'       : 'Mr',
            'Jonkheer' : 'Mr',
            'Lady'     : 'Mrs',
            'Major'    : 'Mr',
            'Mlle'     : 'Miss',
            'Mme'      : 'Mrs',
            'Ms'       : 'Mrs',
            'Rev'      : 'Mr',
            'Sir'      : 'Mr',
        }

    def transform(self, df, y=None):
        df['Title'] = df.apply(self.__get_title, axis=1)
        return df
    
    def __get_title(self, data):
        title = self.__pattern.search(data['Name'].split(',')[1].strip()).group(0).replace('.', '')
        if title == 'Dr':
            return 'Mr' if data['Sex'] == 'male' else 'Mrs'
        else:
            return self.__title_map[title]
class PassengersExtractor(BaseTransformer):
    
    def __init__(self, passengers):
        self.passengers = passengers

    def transform(self, df, y=None):
        df['Passengers'] = df['Ticket'].apply(lambda t: self.passengers[t])
        return df
class SurvivedRatioExtractor(BaseTransformer):

    def __init__(self):
        self.survived = {}
        self.total = {}
        self.fitted_passengers = []
        self.mean = 0

    def fit(self, df, y=None):
        self.survived = df.groupby('Ticket')['Survived'].sum().to_dict()
        self.total = df['Ticket'].value_counts().to_dict()
        self.fitted_passengers = df['PassengerId'].values
        self.mean = df['Survived'].sum() / len(df)
        return self

    def transform(self, df, y=None):
        df['SurvivedRatio'] = df.apply(self.__get_survived_ratio, axis=1)
        return df

    def __get_survived_ratio(self, data):
        survived = self.__get_survived(data)
        total = self.__get_total(data)
        return survived / total if total > 0 else self.mean

    def __get_survived(self, data):
        if data['Ticket'] in self.survived:
            is_fitted = data['PassengerId'] in self.fitted_passengers
            return self.survived[data['Ticket']] - (data['Survived'] if is_fitted else 0)
        else:
            return 0
        
    def __get_total(self, data):
        if data['Ticket'] in self.total:
            is_fitted = data['PassengerId'] in self.fitted_passengers
            return self.total[data['Ticket']] - (1 if is_fitted else 0)
        else:
            return 0        
class OneHotEncoder(BaseTransformer):
    
    def transform(self, df, y=None):
        for column, drop in [('Pclass', '3'), ('Embarked', 'S'), ('Title', 'Master')]:
            df = pd.get_dummies(df, prefix=[column], columns=[column]).drop([column + '_' + drop], axis=1, errors='ignore')
        return df
class FeatureEraser(BaseTransformer):

    def transform(self, df, y=None):
        return df.drop([
            'PassengerId',
            'Survived',
            'Name',
            'Sex',
            'Age',
            'SibSp',
            'Parch',
            'Ticket',
            'Fare',
            'Cabin'
        ], axis=1, errors='ignore')
class DataFrameToArrayConverter(BaseTransformer):

    def transform(self, df, y=None):
        return df.values.astype(float)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_pipeline(classifier):
    passengers = pd.concat([df_train, df_test], ignore_index=True)['Ticket'].value_counts().to_dict()
    return Pipeline([
        ('dup', DataFrameDuplicator()),
        ('ex1', TitleExtractor()),
        ('ex2', PassengersExtractor(passengers)),
        ('ex3', SurvivedRatioExtractor()),
        ('ohe', OneHotEncoder()),
        ('eras', FeatureEraser()),
        ('conv', DataFrameToArrayConverter()),
        ('scl', StandardScaler()),
        ('clf', classifier)
    ])
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipeline = build_pipeline(SVC())
param_grid = [{
    'clf__gamma': np.power(10, np.arange(-2.0, 0.1, 0.1)),
    'clf__C': np.power(10, np.arange(-2.0, 2.1, 0.2))
}]

gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='accuracy', cv=10)

y_train = df_train['Survived'].values.astype(int)
gs.fit(df_train, y_train)

# Best score = 0.851852 (gamma=0.063096, C=6.309573)
best_gamma = gs.best_params_['clf__gamma']
best_C = gs.best_params_['clf__C']
print('Best score = %f (gamma=%f, C=%f)' % (gs.best_score_, best_gamma, best_C))
from sklearn.svm import SVC

# best_gamma = 0.0630957
# best_C = 6.30957

y_train = df_train['Survived'].values.astype(int)

pipeline = build_pipeline(SVC(gamma=best_gamma, C=best_C))
pipeline.fit(df_train, y_train)
predicted = pipeline.predict(df_test)
# Accuracy is 0.808 by the public leaderboard.
submission = pd.DataFrame({ "PassengerId": df_test.loc[:, 'PassengerId'], "Survived": predicted })
submission.to_csv('./submission.csv', index=False)