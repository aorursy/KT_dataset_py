import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize = (16, 12))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title('Skew: ' + str(round(float(df[var_name].skew()),)))
        ax.set_xticklabels([],visible=False)
        ax.set_yticklabels([],visible=False)
    fig.tight_layout()
    plt.show()
    
def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row = row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    
def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()
    
def plot_correlation_map(df):
    corr = titanic.corr()
    _, ax = plt.subplots(figsize=(12,10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={'shrink':.9}, ax=ax, annot=True, annot_kws={'fontsize':12})
    
def describe_more(df):
    var = []; l = []; t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DateFrame({'Variable':var, 'Levels':l,'Datatype':t})
    levels.sort_values(by='Levels', inplace=True)
    return levels

def plot_variable_importance(X,y):
    tree = DecisionTreeClassifier(random_state=99)
    tree.fit(X,y)
    plot_model_var_imp(tree,X,y)
    
def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(model.feature_importances_, columns = ['Importance'], index=X.columns)
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[:10].plot(kind='barh')
    print(model.score(X,y))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

full = train.append(test, ignore_index=True)
titanic = full[:891]

print('Datasets:', 'full:', full.shape, 'titanic:', titanic.shape)
titanic.head()
titanic.describe()
plot_correlation_map(titanic)
plot_distribution(titanic, var='Age', target='Survived', row='Sex')
plot_distribution(titanic, var = 'Fare', target = 'Survived')
plot_categories(titanic, cat='Embarked', target='Survived')
plot_categories(titanic, cat='Sex', target='Survived')
plot_categories(titanic, cat='Pclass', target='Survived')
plot_categories(titanic, cat='SibSp', target='Survived')
plot_categories(titanic, cat='Parch', target='Survived')
sex = pd.Series(np.where(full.Sex=='male', 1, 0), name='Sex')
#?np.where
embarked = pd.get_dummies(full.Embarked, prefix='Emarked')
embarked.head()
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
imputed = pd.DataFrame()
imputed['Age'] = full.Age.fillna(full.Age.mean())
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())
imputed.head()
title = pd.DataFrame()
title['Title'] = full['Name'].map( lambda name: name.split(',')[1].split('.')[0].strip())

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

title['Title'] = title.Title.map(Title_Dictionary)
title = pd.get_dummies(title.Title)
title.head()
cabin = pd.DataFrame()
cabin['Cabin'] = full.Cabin.fillna('U')
cabin['Cabin'] = cabin['Cabin'].map( lambda c:c[0])
cabin = pd.get_dummies(cabin['Cabin'], prefix= 'Cabin')
cabin.head()
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map( lambda t: t.strip(), ticket)
    ticket = list(filter( lambda t : not t.isdigit(), ticket))
    if len(ticket) >0:
        return ticket[0]
    else: 
        return 'XXX'
    
ticket = pd.DataFrame()
ticket['Ticket'] = full['Ticket'].map(cleanTicket)
ticket = pd.get_dummies(ticket['Ticket'], prefix = 'Ticket')
ticket.shape
ticket.head()
family = pd.DataFrame()
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1
family['Family_Single'] = family['FamilySize'].map( lambda s: 1 if s==1 else 0)
family['Family_Small'] = family['FamilySize'].map( lambda s: 1 if 2<=s <=4 else 0)
family['Family_Large'] = family['FamilySize'].map( lambda s: 1 if s>=5 else 0)
family.head()
full_X = pd.concat([imputed, embarked, cabin, sex], axis=1)
full_X.head()
train_valid_X = full_X[0:891]
train_valid_y = titanic.Survived
test_X = full_X[891:]
train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, train_size = .7)
print(full_X.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape)
from sklearn.tree import DecisionTreeClassifier
plot_variable_importance(train_X, train_y)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(train_X, train_y)
print (model.score(train_X, train_y), model.score(valid_X, valid_y))
plot_model_var_imp(model, train_X, train_y)
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold(2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )
test_Y = model.predict(test_X)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.Survived.astype(int)
test.shape
test.head()

#test.to_csv( 'titanic_pred.csv' , index = False )
test['Survived'] = test['Survived'].astype(int)
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )
