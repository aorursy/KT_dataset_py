# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
import seaborn as sns


from matplotlib.ticker import FuncFormatter
# enable static images of your plot embedded in the notebook
%matplotlib inline 

# seaborn
import seaborn as sns
sns.set() #apply the default default seaborn theme, scaling, and color palette

# sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# plotly
import plotly.offline as py
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.express as px
init_notebook_mode(connected=True) # initiate notebook for offline plot

sns.set()

# Graphics in retina format are more sharp and legible
%config InlineBackend.figure_format = 'retina'

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)

df_train.head()
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
df.loc[(df.age >= 60) & (df.occupation == " ?"),'occupation']='senior citizen'

df.loc[(df.age >= 60) & (df.workclass == " ?"),'workclass']='Retired'

df.loc[(df.workclass == ' Never-worked'),'occupation']='Unemployed'

df.loc[(df["age"] <= 24) & (df.workclass == " ?"),'workclass']=' Private'

df.loc[(df["age"] <= 21) & (df.occupation == " ?") & (df.education == " Some-college"),'occupation']='part-time'
df.loc[(df["age"] <= 21) & (df.occupation == " ?") & (df.education == " HS-grad") & (df['hours-per-week'] < 40),'occupation']='part-time'
df.loc[(df["age"] <= 21) & (df.occupation == " ?") & (df.education == " HS-grad") & (df['hours-per-week'] >= 40),'occupation']='full-time'

df.loc[(df["age"] <= 24) & (df.occupation == " ?") & (df.education == " Some-college") & (df['hours-per-week'] >= 40),'occupation']='full-time'
df.loc[(df["age"] <= 24) & (df.occupation == " ?") & (df.education == " Some-college") & (df['hours-per-week'] < 40),'occupation']='part-time'

df.loc[(df.workclass == " ?"),'workclass']=' Private'
df.loc[(df.occupation == " ?") & (df['native-country'] != ' United-States') & (df['hours-per-week'] <= 25),'occupation']='part-time'
df['married']=df['marital-status'].apply(lambda x: 1 if x.strip().startswith("Married") else 0 )
df = df.drop(['marital-status'], axis=1)
df_temp1 = df.loc[
    df['target'].notna()
].groupby(
    ['workclass']
)[
    'target'
].agg(['mean']).rename(
    columns={'mean': 'WCtarget_mean'}
).fillna(0.0).reset_index()
           
             
df_temp2 = df.loc[
    df['target'].notna()
].groupby(
    ['education']
)[
    'target'
].agg(['mean']).rename(
    columns={'mean': 'edutarget_mean'}
).fillna(0.0).reset_index()           

df_temp3 = df.loc[
    df['target'].notna()
].groupby(
    ['occupation']
)[
    'target'
].agg(['mean']).rename(
    columns={'mean': 'octarget_mean'}
).fillna(0.0).reset_index()


df_temp4 = df.loc[
    df['target'].notna()
].groupby(
    ['race']
)[
    'target'
].agg(['mean']).rename(
    columns={'mean': 'racetarget_mean'}
).fillna(0.0).reset_index()


df_temp5 = df.loc[
    df['target'].notna()
].groupby(
    ['native-country']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'NCtarget_mean', 'std': 'NCtarget_std'}
).fillna(0.0).reset_index()

df_temp6 = df.loc[
    df['target'].notna()
].groupby(
    ['relationship']
)[
    'target'
].agg(['mean']).rename(
    columns={'mean': 'relatarget_mean'}
).fillna(0.0).reset_index()

df = pd.merge(df, 
                  df_temp1,
                  how = 'left',
                  on=['workclass']
                 )

df = pd.merge(df, 
                  df_temp2,
                  how = 'left',
                  on=['education']
                 )
df = pd.merge(df, 
                  df_temp3,
                  how = 'left',
                  on=['occupation']
                 )
df = pd.merge(df, 
                  df_temp4,
                  how = 'left',
                  on=['race']
                 )
df = pd.merge(df, 
                  df_temp5,
                  how = 'left',
                  on=['native-country']
                 )

df = pd.merge(df, 
                  df_temp6,
                  how = 'left',
                  on=['relationship']
                 )
df.drop(columns=['education','education-num', 'race', 'occupation','workclass','native-country','relationship'],inplace=True)
df.head()
df['WCtarget_mean'] = df['WCtarget_mean'].fillna(0.0)
df['edutarget_mean'] = df['edutarget_mean'].fillna(0.0)
df['octarget_mean'] = df['octarget_mean'].fillna(0.0)
df['racetarget_mean'] = df['racetarget_mean'].fillna(0.0)
df['NCtarget_mean'] = df['NCtarget_mean'].fillna(0.0)
df['NCtarget_std'] = df['NCtarget_std'].fillna(0.0)
df['relatarget_mean'] = df['relatarget_mean'].fillna(0.0)
df = pd.get_dummies(
    df, 
    columns=[c for c in df.columns if df[c].dtype == 'object']
)
df.head()
our_x_train = df.loc[df['target'].notna()].drop(columns=['target'])
our_y_train = df.loc[df['target'].notna()]['target']
our_x_test = df.loc[df['target'].isna()].drop(columns=['target'])
our_y_test = df.loc[df['target'].isna()]['target']
from sklearn.tree import DecisionTreeClassifier

X_train, X_holdout, y_train, y_holdout = train_test_split(our_x_train,
                                                    our_y_train,
                                                    test_size=0.33,
                                                    random_state=17)
tree = DecisionTreeClassifier(max_depth= 8, random_state=17,max_features= 9)

tree.fit(X_train, y_train)

tree = tree.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])

parameter_grid = {
    'max_depth': range(5, 15),
    'max_features': range(1, 9),
    'min_samples_split': [35, 40, 45, 50],
    'min_samples_leaf': [5, 10, 15, 20],
}
grid_search = GridSearchCV(tree, param_grid=parameter_grid, cv=5)
grid_search.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

tree_pred = tree.predict(X_holdout)
accuracy_score(y_holdout, tree_pred)
tree.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
p = tree.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': p
})
from sklearn.metrics import log_loss
log_loss(y_holdout, grid_search.predict_proba(X_holdout)[:, 1])
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
!head /kaggle/working/submit.csv