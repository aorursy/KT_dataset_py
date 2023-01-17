# Loading libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import xgboost as xgb

import statsmodels.api as sm

from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from statsmodels.formula.api import ols





# Import datasets

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')

# Dimension of train dataset

train.shape
# Dimension of test dataset

test.shape
# Target variable

y = train.Cover_Type

train.drop(['Cover_Type'], axis=1, inplace=True)
# Whole dataframe

df = pd.concat([train,test], axis=0)

X_full = df
# dimensions of dataset

print(X_full.shape)

# list types for each attribute

X_full.dtypes

# take a peek at the first rows of the data

X_full.head(5)

# summarize attribute distributions for data frame

print(X_full.describe().T)

print(X_full.info())

def rstr(X_full): return X_full.shape, X_full.apply(lambda x: [x.unique()])

print(rstr(X_full))

# Look at the level of each feature

for column in X_full.columns:

    print(column, X_full[column].nunique())
# numerical features

X_full['Elevation'] = X_full['Elevation'].astype(float)

X_full['Aspect'] = X_full['Aspect'].astype(float)

X_full['Slope'] = X_full['Slope'].astype(float)

X_full['Horizontal_Distance_To_Hydrology'] = X_full['Horizontal_Distance_To_Hydrology'].astype(float)

X_full['Vertical_Distance_To_Hydrology'] = X_full['Vertical_Distance_To_Hydrology'].astype(float)

X_full['Horizontal_Distance_To_Roadways'] = X_full['Horizontal_Distance_To_Roadways'].astype(float)

X_full['Hillshade_9am'] = X_full['Hillshade_9am'].astype(float)

X_full['Hillshade_Noon'] = X_full['Hillshade_Noon'].astype(float)

X_full['Hillshade_3pm'] = X_full['Hillshade_3pm'].astype(float)

X_full['Horizontal_Distance_To_Fire_Points'] = X_full['Horizontal_Distance_To_Fire_Points'].astype(float)

# check missing values both to numeric features and categorical features 

feat_missing = []



for f in X_full.columns:

    missings = X_full[f].isnull().sum()

    if missings > 0:

        feat_missing.append(f)

        missings_perc = missings/X_full.shape[0]

        

        # printing summary of missing values

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))



# how many variables do present missing values?

print()

print('In total, there are {} variables with missing values'.format(len(feat_missing)))
# summarize the class distribution

y = y.astype(object) 

count = pd.crosstab(index = y, columns="count")

percentage = pd.crosstab(index = y, columns="frequency")/pd.crosstab(index = y, columns="frequency").sum()

pd.concat([count, percentage], axis=1)
ax = sns.countplot(x=y, data=X_full).set_title("Target Variable Distribution")
# categorical features

categorical_cols = [cname for cname in X_full.columns if

                    X_full[cname].dtype in ['int64']]

cat = X_full[categorical_cols]

cat.columns
cat_train = cat.iloc[0:15119,:]

cat_test = cat.iloc[15120:565892,:]
def rstr(cat_train): return cat_train.shape, cat_train.apply(lambda x: [x.unique()])

print(rstr(cat_train))

def rstr(cat_test): return cat_test.shape, cat_test.apply(lambda x: [x.unique()])

print(rstr(cat_test))
# Drop features not helpful 

cat = cat.drop(['Id', 'Soil_Type15', 'Soil_Type7'], axis=1)
# Visualizations

sns.set( rc = {'figure.figsize': (5, 5)})

fcat = ['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type8','Soil_Type9',

       'Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type16','Soil_Type17','Soil_Type18',

        'Soil_Type19','Soil_Type20', 'Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26',

        'Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34',

        'Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Wilderness_Area1',

       'Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']



for col in fcat:

    plt.figure()

    sns.countplot(x=cat[col], data=cat, palette="Set3")

    plt.show()
# Chi-Squared test as Feature Selection

cat2 = pd.concat([y,cat_train], axis=1)

class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi-2 Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from the model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)





# Initialize Chi-Squared Test

cT = ChiSquare(cat2)



# Feature Selection

testColumns = ['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type8','Soil_Type9',

       'Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type16','Soil_Type17','Soil_Type18',

        'Soil_Type19','Soil_Type20', 'Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26',

        'Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34',

        'Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Wilderness_Area1',

       'Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']



for var in testColumns:

    cT.TestIndependence(colX=var,colY='Cover_Type') 
# Drop feature not helpful by Feature Selection

cat = cat.drop(['Soil_Type8', 'Soil_Type25'], axis=1)

cat.shape
# Numerical features

numerical_cols = [cname for cname in X_full.columns if

                 X_full[cname].dtype in ['float']]

num = X_full[numerical_cols]

num.columns
def rstr(num): return num.shape, num.apply(lambda x: [x.nunique()])

print(rstr(num))
# Visualizations

sns.set( rc = {'figure.figsize': (5, 5)})

fnum = ['Aspect','Elevation', 'Hillshade_3pm','Hillshade_9am', 'Hillshade_Noon', 'Horizontal_Distance_To_Fire_Points', 

        'Horizontal_Distance_To_Hydrology','Horizontal_Distance_To_Roadways', 'Slope', 'Vertical_Distance_To_Hydrology']



for col in fnum:

    plt.figure()

    x=num[col]

    sns.distplot(x, bins=10)

    plt.xticks(rotation=45)

    plt.show()      
for col in fnum:

    plt.figure()

    x=num[col]

    sns.boxplot(x,palette="Set1",linewidth=1)

    plt.xticks(rotation=45)

    plt.show()  
# Anova Test as Feature Selection

num_train = num.iloc[0:15119,:]

num_test = num.iloc[15120:565892,:]

num2 = pd.concat([y,num_train], axis=1)

num2['Cover_Type'] = num2['Cover_Type'].astype(int)
results = ols('Cover_Type ~ Aspect+Elevation+Hillshade_3pm+Hillshade_9am+Hillshade_Noon+Horizontal_Distance_To_Fire_Points+Horizontal_Distance_To_Hydrology+Horizontal_Distance_To_Roadways+Slope+Vertical_Distance_To_Hydrology', data=num2).fit()

aov_table = sm.stats.anova_lm(results, typ=2)

aov_table
# Outliers



# Elevation

#before

plt.figure()

x=num['Elevation']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Elevation'

q75, q25 = np.percentile(num.Elevation.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Elevation.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Elevation']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()







# Hillshade_3pm

#before

plt.figure()

x=num['Hillshade_3pm']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show() 



#corrections

i = 'Hillshade_3pm'

q75, q25 = np.percentile(num.Hillshade_3pm.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Hillshade_3pm.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Hillshade_3pm']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show() 





# Hillshade_9am

#before

plt.figure()

x=num['Hillshade_9am']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Hillshade_9am'

q75, q25 = np.percentile(num.Hillshade_9am.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Hillshade_9am.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Hillshade_9am']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()





# Hillshade_Noon

#before

plt.figure()

x=num['Hillshade_Noon']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Hillshade_Noon'

q75, q25 = np.percentile(num.Hillshade_Noon.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Hillshade_Noon.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Hillshade_Noon']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()





# Horizontal_Distance_To_Fire_Points

#before

plt.figure()

x=num['Horizontal_Distance_To_Fire_Points']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Horizontal_Distance_To_Fire_Points'

q75, q25 = np.percentile(num.Horizontal_Distance_To_Fire_Points.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Horizontal_Distance_To_Fire_Points.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Horizontal_Distance_To_Fire_Points']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



# Horizontal_Distance_To_Hydrology

#before

plt.figure()

x=num['Horizontal_Distance_To_Hydrology']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Horizontal_Distance_To_Hydrology'

q75, q25 = np.percentile(num.Horizontal_Distance_To_Hydrology.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Horizontal_Distance_To_Hydrology.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Horizontal_Distance_To_Hydrology']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()





# Horizontal_Distance_To_Roadways

#before

plt.figure()

x=num['Horizontal_Distance_To_Roadways']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Horizontal_Distance_To_Roadways'

q75, q25 = np.percentile(num.Horizontal_Distance_To_Roadways.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Horizontal_Distance_To_Roadways.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Horizontal_Distance_To_Roadways']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()





# Slope

#before

plt.figure()

x=num['Slope']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Slope'

q75, q25 = np.percentile(num.Slope.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Slope.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Slope']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()





# Vertical_Distance_To_Hydrology

#before

plt.figure()

x=num['Vertical_Distance_To_Hydrology']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()



#corrections

i = 'Vertical_Distance_To_Hydrology'

q75, q25 = np.percentile(num.Vertical_Distance_To_Hydrology.dropna(), [75 ,25])

q95, q05 = np.percentile(num.Vertical_Distance_To_Hydrology.dropna(), [95 ,5])

iqr = q75 - q25 

min = q25 - (iqr*1.5)

max = q75 + (iqr*1.5) 

num[i].loc[num[i] < min] = q05

num[i].loc[num[i] > max] = q95



#after

plt.figure()

x=num['Vertical_Distance_To_Hydrology']

sns.boxplot(x,palette="Set2",linewidth=1)

plt.xticks(rotation=45)

plt.show()

num['X1X2'] = num['Aspect']*num['Elevation']

num['X1X3'] = num['Aspect']*num['Hillshade_3pm']

num['X1X4'] = num['Aspect']*num['Hillshade_9am']

num['X1X5'] = num['Aspect']*num['Hillshade_Noon']

num['X1X6'] = num['Aspect']*num['Horizontal_Distance_To_Fire_Points']

num['X1X7'] = num['Aspect']*num['Horizontal_Distance_To_Hydrology']

num['X1X8'] = num['Aspect']*num['Horizontal_Distance_To_Roadways']

num['X1X9'] = num['Aspect']*num['Slope']

num['X1X10'] = num['Aspect']*num['Vertical_Distance_To_Hydrology']

num['X2X3'] = num['Elevation']*num['Hillshade_3pm']

num['X2X4'] = num['Elevation']*num['Hillshade_9am']

num['X2X5'] = num['Elevation']*num['Hillshade_Noon']

num['X2X6'] = num['Elevation']*num['Horizontal_Distance_To_Fire_Points']

num['X2X7'] = num['Elevation']*num['Horizontal_Distance_To_Hydrology']

num['X2X8'] = num['Elevation']*num['Horizontal_Distance_To_Roadways']

num['X2X9'] = num['Elevation']*num['Slope']

num['X2X10'] = num['Elevation']*num['Vertical_Distance_To_Hydrology']

num['X3X4'] = num['Hillshade_3pm']*num['Hillshade_9am']

num['X3X5'] = num['Hillshade_3pm']*num['Hillshade_Noon']

num['X3X6'] = num['Hillshade_3pm']*num['Horizontal_Distance_To_Fire_Points']

num['X3X7'] = num['Hillshade_3pm']*num['Horizontal_Distance_To_Hydrology']

num['X3X8'] = num['Hillshade_3pm']*num['Horizontal_Distance_To_Roadways']

num['X3X9'] = num['Hillshade_3pm']*num['Slope']

num['X3X10'] = num['Hillshade_3pm']*num['Vertical_Distance_To_Hydrology']

num['X4X5'] = num['Hillshade_9am']*num['Hillshade_Noon']

num['X4X6'] = num['Hillshade_9am']*num['Horizontal_Distance_To_Fire_Points']

num['X4X7'] = num['Hillshade_9am']*num['Horizontal_Distance_To_Hydrology']

num['X4X8'] = num['Hillshade_9am']*num['Horizontal_Distance_To_Roadways']

num['X4X9'] = num['Hillshade_9am']*num['Slope']

num['X4X10'] = num['Hillshade_9am']*num['Vertical_Distance_To_Hydrology']

num['X5X6'] = num['Hillshade_Noon']*num['Horizontal_Distance_To_Fire_Points']

num['X5X7'] = num['Hillshade_Noon']*num['Horizontal_Distance_To_Hydrology']

num['X5X8'] = num['Hillshade_Noon']*num['Horizontal_Distance_To_Roadways']

num['X5X9'] = num['Hillshade_Noon']*num['Slope']

num['X5X10'] = num['Hillshade_Noon']*num['Vertical_Distance_To_Hydrology']

num['X6X7'] = num['Horizontal_Distance_To_Fire_Points']*num['Horizontal_Distance_To_Hydrology']

num['X6X8'] = num['Horizontal_Distance_To_Fire_Points']*num['Horizontal_Distance_To_Roadways']

num['X6X9'] = num['Horizontal_Distance_To_Fire_Points']*num['Slope']

num['X6X10'] = num['Horizontal_Distance_To_Fire_Points']*num['Vertical_Distance_To_Hydrology']

num['X7X8'] = num['Horizontal_Distance_To_Hydrology']*num['Horizontal_Distance_To_Roadways']

num['X7X9'] = num['Horizontal_Distance_To_Hydrology']*num['Slope']

num['X7X10'] = num['Horizontal_Distance_To_Hydrology']*num['Vertical_Distance_To_Hydrology']

num['X8X9'] = num['Horizontal_Distance_To_Roadways']*num['Slope']

num['X8X10'] = num['Horizontal_Distance_To_Roadways']*num['Vertical_Distance_To_Hydrology']

num['X9X10'] = num['Slope']*num['Vertical_Distance_To_Hydrology']

X_all = pd.concat([cat, num], axis=1)

X_all.shape
# Create correlation matrix

corr_matrix = X_all.corr().abs()

# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.75

to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

to_drop
# Drop features 

X_all.drop(X_all[to_drop], axis=1, inplace=True)

X_all.shape

y = y.astype('int')

train_ = X_all.iloc[0:15120,:]

test_ = X_all.iloc[15121:581012,:]

# Break off train and validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(train_, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

# Test options and evaluation metric



# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression(random_state=0)))

models.append(('BAG', BaggingClassifier(random_state=0)))

models.append(('RF', RandomForestClassifier(random_state=0)))

models.append(('ADA', AdaBoostClassifier(random_state=0)))

models.append(('GBM', GradientBoostingClassifier(random_state=0)))

models.append(('XGB', XGBClassifier(random_state=0)))

results_t = []

results_v = []

names = []

score = []

for name, model in models:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=5)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    score.append(f_dict)

    

score = pd.DataFrame(score, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score)
# Spot Check Algorithms with standardized dataset

pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression(random_state=0))])))

pipelines.append(('ScaledBAG', Pipeline([('Scaler', StandardScaler()),('BAG', BaggingClassifier(random_state=0))])))

pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier(random_state=0))])))

pipelines.append(('ScaledADA', Pipeline([('Scaler', StandardScaler()),('ADA', AdaBoostClassifier(random_state=0))])))

pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier(random_state=0))])))

pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB', XGBClassifier(random_state=0))])))

pipelines.append(('ScaledNN', Pipeline([('Scaler', StandardScaler()),('NN', MLPClassifier(random_state=0))])))

results_t = []

results_v = []

names = []

score_sd = []

for name, model in pipelines:

    param_grid = {}

    my_model = GridSearchCV(model,param_grid,cv=5)

    my_model.fit(X_train, y_train)

    predictions_t = my_model.predict(X_train) 

    predictions_v = my_model.predict(X_valid)

    accuracy_train = accuracy_score(y_train, predictions_t) 

    accuracy_valid = accuracy_score(y_valid, predictions_v) 

    results_t.append(accuracy_train)

    results_v.append(accuracy_valid)

    names.append(name)

    f_dict = {

        'model': name,

        'accuracy_train': accuracy_train,

        'accuracy_valid': accuracy_valid,

    }

    score_sd.append(f_dict)

    

score_sd = pd.DataFrame(score_sd, columns = ['model','accuracy_train', 'accuracy_valid'])
print(score_sd)
model = RandomForestClassifier()

model.fit(X_train, y_train)



(pd.Series(model.feature_importances_, index=X_train.columns)

   .nlargest(4)

   .plot(kind='barh')).set_title("Random Forest") 
model = AdaBoostClassifier()

model.fit(X_train, y_train)



(pd.Series(model.feature_importances_, index=X_train.columns)

   .nlargest(4)

   .plot(kind='barh')).set_title("AdaBoost") 
model = GradientBoostingClassifier()

model.fit(X_train, y_train)



(pd.Series(model.feature_importances_, index=X_train.columns)

   .nlargest(4)

   .plot(kind='barh')).set_title("Gradient Boosting") 
model = XGBClassifier()

model.fit(X_train, y_train)



(pd.Series(model.feature_importances_, index=X_train.columns)

   .nlargest(4)

   .plot(kind='barh')).set_title("XGBoost") 