# 1.0 Call libraries



# Data manipulation

%reset -f

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

import os



# Set a few plotting defaults

%matplotlib inline



plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['patch.edgecolor'] = 'k'



# 1.0.1 For measuring time elapsed

import time



from collections import OrderedDict
# 1.1 Working with imbalanced data

# http://contrib.scikit-learn.org/imbalanced-learn/stable/generated/imblearn.over_sampling.SMOTE.html

# Check imblearn version number as:

#   import imblearn;  imblearn.__version__

from imblearn.over_sampling import SMOTE, ADASYN



# 1.2 Processing data

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import  OneHotEncoder as ohe

from sklearn.preprocessing import StandardScaler as ss

from sklearn.compose import ColumnTransformer as ct



# 1.3 Data imputation

from sklearn.impute import SimpleImputer

# 1.4 Model building

from sklearn.linear_model import LogisticRegression



# 1.5 for ROC graphs & metrics

import scikitplot as skplt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import average_precision_score

import sklearn.metrics as metrics



# to make this notebook's output stable across runs

#Somehow this is not happening as o/p of models is not consistent

np.random.seed(42)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

from sklearn.exceptions import ConvergenceWarning



# Filter out warnings from models



warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

warnings.filterwarnings('ignore', category = ConvergenceWarning)

warnings.filterwarnings('ignore', category = DeprecationWarning)

warnings.filterwarnings('ignore', category = UserWarning)

warnings.filterwarnings('ignore', category = FutureWarning)



# 1.9 Misc

import gc
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline





# Custom scorer for cross validation

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
# 1.3 Dimensionality reduction

from sklearn.decomposition import PCA



# 1.4 Data splitting and model parameter search

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from bayes_opt import BayesianOptimization



# 1.5 Modeling modules

# conda install -c anaconda py-xgboost

from xgboost.sklearn import XGBClassifier
# Model imports

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import ExtraTreesClassifier
pd.options.display.max_columns = 150



# Read in data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



#train = pd.read_csv('train.csv')

#test = pd.read_csv('test.csv')

train.head()

# 3.0 Let us understand train data

# 3.1 Begin by defining some functions

def ExamineData(x):

    """Prints various data charteristics, given x

    """

    print("Data shape:", x.shape)

    print("\nColumns:", x.columns)

    print("\nData types\n", x.dtypes)

    print("\nDescribe data\n", x.describe())

    print("\nData\n", x.head(2))

    print ("\nSize of data:", np.sum(x.memory_usage()))    # Get size of dataframes

    print("\nAre there any NULLS\n", np.sum(x.isnull()))

# 3.2 start examining data - commented after analysis due to large data dump on screen.

#ExamineData(train)
# commented after analysis due to large data dump on screen.

#ExamineData(test)
def PlotKDE(x):

    



    plt.figure(figsize = (20, 15))

#    plt.style.use('fivethirtyeight')

#    plt.style.available

    plt.style.use('seaborn-pastel')



    # Color mapping

    colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})

    poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})



    # Iterate through the columns

    for i, col in enumerate(x):

        ax = plt.subplot(8, 5, i + 1)

        # Iterate through the poverty levels

        for poverty_level, color in colors.items():

            # Plot each poverty level as a separate line

            sns.kdeplot(train.loc[train['Target'] == poverty_level, col].dropna(), 

                        ax = ax, color = color, label = poverty_mapping[poverty_level])

        

        plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



    plt.subplots_adjust(top = 2)

#PlotKDE(train.select_dtypes('int64'))
# 3.4 Visual examination of float columns

PlotKDE(train.select_dtypes('float'))
# 3.5 Object data types



train.select_dtypes('object').head()



mapObj = {"yes": 1, "no": 0}
# Apply same operation to both train and test

for df in [train, test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapObj).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapObj).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapObj).astype(np.float64)



train[['dependency', 'edjefa', 'edjefe']].describe()
PlotKDE(train.select_dtypes('float')) # the parameters are now classified as float
# 4.1 filling up column Target in test with nan

test['Target'] = np.nan



#4.2 appending test to train

X = train.append(test, ignore_index = True)
#4.3.1 Shape

train.shape #(9557, 143)

test.shape #(23856, 143)

X.shape #Sum of test and train: (33413, 143)
#4.3.2 info

train.info()

test.info()

X.info() 
#4.4 Exploring Data distribution across classes

#4.4.1 Extract the records for heads of household where 'parentesco1==1'

X_heads = X.loc[X['parentesco1']==1].copy() #Make a copy to preserve X

X_heads.info() #10307 entries, 0 to 33409

#4.4.2 look at label distribution where 'Target is notnull'

X_heads_labels = X_heads.loc[(X_heads['Target'].notnull()), ['Target']]

X_heads_labels_counts = X_heads_labels['Target'].value_counts().sort_index()

X_heads_labels_counts 
#4.5.1 Grouping by headh of household 'idhogar' and adding 

#train.groupby('idhogar').size() #Length: 2988

train_ok = train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)                    

train_ok.size #2988 households
#4.5.2 Identify the labels with errors

train_notok = train_ok[train_ok != True]

train_notok.size #85 labels have errors
#4.5.3 View one example of incorrect labels

train[train['idhogar'] == train_notok.index[2]][['Id', 'idhogar', 'parentesco1', 'Target']]

#4.5.4 Fix the labels correctly



for not_ok_id in train_notok.index:

    # Find correct Target value for head of household

    # not_ok_id

    ok_target = int(train[(train['idhogar'] == not_ok_id) & (train['parentesco1'] == 1.0)]['Target'])

    

    # Set the correct label for all members in the household

    train.loc[train['idhogar'] == not_ok_id, 'Target'] = ok_target
#Checking - Trying query function of dataframe

train_check = pd.DataFrame(train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1))

train_check.query("Target != True") #Empty DataFrame here is a Success!
#4.6.1 how many huseholds have parentesco1

train_heads = pd.DataFrame(train.groupby('idhogar')['parentesco1'].sum())

train_heads.size #2988 records

train_heads.query("parentesco1 > 1").count() #just checking -- 0 households have more than one head of household

train_heads.query("parentesco1 == 1").count() #just checking -- 2973 households OK



train_heads.query("parentesco1 < 1").count() #15 households do not have head of household

#4.6.2 How many of the households have sum(parentesco1) computed in 4.6.1 as zero

""" Cannot use these households data """

train_heads_no = train_heads.query("parentesco1 == 0") #15 unique 'idhogar's

# Number of missing in each column

missing = pd.DataFrame(X.isnull().sum()).rename(columns = {0: 'total'})



# Create a percentage missing

missing['percent'] = missing['total'] / len(X)



missing.sort_values('percent', ascending = False).head(7).drop('Target')
#X_heads['v18q1'].value_counts().sort_index()

X_heads['v18q1'].value_counts()
X_heads['v18q1'].value_counts().sum()
X_heads['v18q1'].isnull().describe()
X_heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
X['v18q1'] = X['v18q1'].fillna(0)
# Fill in households that own the house with 0 rent payment

X.loc[(X['tipovivi1'] == 1), 'v2a1'] = 0



# Create missing rent payment column

X['v2a1-missing'] = X['v2a1'].isnull()



X['v2a1-missing'].value_counts()
X.loc[X['rez_esc'].notnull()]['age'].describe()
X.loc[X['rez_esc'].isnull()]['age'].describe()
# If individual is over 19 or younger than 7 and missing years behind, set it to 0

X.loc[((X['age'] > 19) | (X['age'] < 7)) & (X['rez_esc'].isnull()), 'rez_esc'] = 0



# Add a flag for those between 7 and 19 with a missing value

X['rez_esc-missing'] = X['rez_esc'].isnull()
X.loc[X['rez_esc'] > 5, 'rez_esc'] = 5
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone', 'rez_esc-missing']



ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2', 'v2a1-missing']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_

# Remove squared variables

X = X.drop(columns = sqr_)

X.shape #(33413, 136)
heads = X.loc[X['parentesco1'] == 1, :]

heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

heads.shape
# Owns a refrigerator, computer, tablet, and television

heads['bonus'] = 1 * (heads['refrig'] + 

                      heads['computer'] + 

                      (heads['v18q1'] > 0) + 

                      heads['television'])



sns.violinplot('bonus', 'Target', data = heads,

                figsize = (10, 6));

plt.title('Target vs Bonus Variable');
heads['phones-per-capita'] = heads['qmobilephone'] / heads['tamviv']

heads['tablets-per-capita'] = heads['v18q1'] / heads['tamviv']

heads['rooms-per-capita'] = heads['rooms'] / heads['tamviv']

heads['rent-per-capita'] = heads['v2a1'] / heads['tamviv']
household_feats = list(heads.columns)
ind = X[id_ + ind_bool + ind_ordered]

ind.shape #(33413, 40)
ind = ind.drop(columns = 'male')
ind['escolari/age'] = ind['escolari'] / ind['age']



plt.figure(figsize = (10, 8))

sns.violinplot('Target', 'escolari/age', data = ind);
# Group and aggregate

ind_agg = ind.drop(columns = 'Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std'])

ind_agg.head()
# Rename the columns

new_col = []

for c in ind_agg.columns.levels[0]:

    for stat in ind_agg.columns.levels[1]:

        new_col.append(f'{c}-{stat}')

        

ind_agg.columns = new_col

ind_agg.head()
ind_agg.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].head()
ind_feats = list(ind_agg.columns)



# Merge on the household id

final = heads.merge(ind_agg, on = 'idhogar', how = 'left')



print('Final features shape: ', final.shape)
final.head() #289 columns
head_gender = ind.loc[ind['parentesco1'] == 1, ['idhogar', 'female']]

final = final.merge(head_gender, on = 'idhogar', how = 'left').rename(columns = {'female': 'female-head'})
final.groupby('female-head')['Target'].value_counts(normalize=True)
sns.violinplot(x = 'female-head', y = 'Target', data = final);

plt.title('Target by Female Head of Household');
# Labels for training

train_labels = np.array(list(final[final['Target'].notnull()]['Target'].astype(np.uint8)))



# Extract the training data

train_set = final[final['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

test_set = final[final['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



# Submission base which is used for making submissions to the competition

submission_base = test[['Id', 'idhogar']].copy()
features = list(train_set.columns)



pipeline = Pipeline([('imputer', Imputer(strategy = 'median')), 

                      ('scaler', MinMaxScaler())])



# Fit and transform training data

train_set = pipeline.fit_transform(train_set)

test_set = pipeline.transform(test_set)
len(features)
model_RF = RandomForestClassifier(n_estimators=100, random_state=10, 

                               n_jobs = -1)

# 10 fold cross validation

cv_score_RF = cross_val_score(model_RF, train_set, train_labels, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score_RF.mean(), 4)} with std = {round(cv_score_RF.std(), 4)}')
model_LRL2 = LogisticRegression(C=0.1, penalty='l2', random_state=10, n_jobs = -1)

# 10 fold cross validation

cv_score_LRL2 = cross_val_score(model_LRL2, train_set, train_labels, cv = 10, scoring = scorer)



print(f'10 Fold Cross Validation F1 Score = {round(cv_score_LRL2.mean(), 4)} with std = {round(cv_score_LRL2.std(), 4)}')
model_RF.fit(train_set, train_labels)



# Feature importances into a dataframe

feature_importances = pd.DataFrame({'feature': features, 'importance': model_RF.feature_importances_})

feature_importances.head()
def plot_feature_importances(df, n = 15, threshold = 0.95):

    """Plots n most important features. Also plots the cumulative importance 

    

    Args:

        df (dataframe): Dataframe of feature importances. Columns must be "feature" and "importance".

    

        n (int): Number of most important features to plot. Default is 15.

    

        threshold (float): Threshold for cumulative importance plot. Default is 95%.

        

    Returns:

        df (dataframe): Dataframe ordered by feature importances with a normalized column (sums to 1) 

                        and a cumulative importance column

    

    Note:

    

        * Normalization in this case means sums to 1. 

        * Cumulative importance is calculated by summing features from most to least important

        * A threshold of 0.95 will show the most important features needed to reach 95% of cumulative importance

    

    """

    plt.style.use('fivethirtyeight')

    

    # Sort features with most important at the head

    df = df.sort_values('importance', ascending = False).reset_index(drop = True)

    

    # Normalize the feature importances to add up to one and calculate cumulative importance

    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    

    plt.rcParams['font.size'] = 12

    

    # Bar plot of n most important features

    df.loc[:n, :].plot.barh(y = 'importance_normalized', 

                            x = 'feature', color = 'darkgreen', 

                            edgecolor = 'k', figsize = (12, 8),

                            legend = False, linewidth = 2)



    plt.xlabel('Normalized Importance', size = 18); plt.ylabel(''); 

    plt.title(f'{n} Most Important Features', size = 18)

    plt.gca().invert_yaxis()

    

    

    if threshold:

        # Cumulative importance plot

        plt.figure(figsize = (8, 6))

        plt.plot(list(range(len(df))), df['cumulative_importance'], 'b-')

        plt.xlabel('Number of Features', size = 16); plt.ylabel('Cumulative Importance', size = 16); 

        plt.title('Cumulative Feature Importance', size = 18);

        

        # Number of features needed for threshold cumulative importance

        # This is the index (will need to add 1 for the actual number)

        importance_index = np.min(np.where(df['cumulative_importance'] > threshold))

        

        # Add vertical line to plot

        plt.vlines(importance_index + 1, ymin = 0, ymax = 1.05, linestyles = '--', colors = 'red')

        plt.show();

        

        print('{} features required for {:.0f}% of cumulative importance.'.format(importance_index + 1, 

                                                                                  100 * threshold))

    

    return df

norm_fi = plot_feature_importances(feature_importances, threshold=0.95)
# Dataframe to hold results

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])



model_results = model_results.append(pd.DataFrame({

    'model': 'RandomForestClassifier', 

    'cv_mean': cv_score_RF.mean(), 

    'cv_std': cv_score_RF.std()}, 

    index = [0]),

                                     ignore_index = True)
model_results = model_results.append(pd.DataFrame({

    'model': 'LogisticRegression', 

    'cv_mean': cv_score_LRL2.mean(), 

    'cv_std': cv_score_LRL2.std()}, 

    index = [0]),

                                     ignore_index = True)


def cv_model(train, train_labels, model, name, model_results=None):

    """Perform 10 fold cross validation of a model"""

    

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)

    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')

    

    if model_results is not None:

        model_results = model_results.append(pd.DataFrame({'model': name, 

                                                           'cv_mean': cv_scores.mean(), 

                                                            'cv_std': cv_scores.std()},

                                                           index = [0]),

                                             ignore_index = True)



        return model_results
model_results = cv_model(train_set, train_labels, LinearSVC(), 

                         'LSVC', model_results)
model_results = cv_model(train_set, train_labels, 

                         GaussianNB(), 'GNB', model_results)
model_results = cv_model(train_set, train_labels, 

                         MLPClassifier(hidden_layer_sizes=(32, 64, 128, 64, 32)),

                         'MLP', model_results)
model_results = cv_model(train_set, train_labels, 

                          LinearDiscriminantAnalysis(), 

                          'LDA', model_results)
model_results = cv_model(train_set, train_labels, 

                         RidgeClassifierCV(), 'RIDGE', model_results)
for n in [5, 10, 20]:

    print(f'\nKNN with {n} neighbors\n')

    model_results = cv_model(train_set, train_labels, 

                             KNeighborsClassifier(n_neighbors = n),

                             f'knn-{n}', model_results)
model_results = cv_model(train_set, train_labels, 

                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),

                         'EXT', model_results)
model_results.set_index('model', inplace = True)

model_results['cv_mean'].plot.bar(color = 'aqua', figsize = (8, 6),

                                  yerr = list(model_results['cv_std']),

                                  edgecolor = 'k', linewidth = 2)

plt.title('Model F1 Score Results');

plt.ylabel('Mean F1 Score (with error bar)');

model_results.reset_index(inplace = True)
train_set = pd.DataFrame(train_set, columns = features)

train_set.info()
test_set = pd.DataFrame(test_set, columns = features)

test_set.info()
features = list(train_set.columns)
############### GG. Tuning using Bayes Optimization ############

"""

11. Step 1: Define BayesianOptimization function.

"""

# 11.1 Which parameters to consider and what is each one's range

para_set = {

           'learning_rate':  (0, 1),                 # any value between 0 and 1

           'n_estimators':   (10,100),               # any number between 50 to 300

           'max_depth':      (6,20),                 # any depth between 3 to 10

           'n_components' :  (150,200)               # any number between 150 to 190

            }



# 11.2 Create a function that when passed some parameters

#    evaluates results using cross-validation

#    This function is used by BayesianOptimization() object



def xg_eval(learning_rate,n_estimators, max_depth,n_components):

    # 12.1 Make pipeline. Pass parameters directly here

    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?

                              PCA(n_components=int(round(n_components))),

                              XGBClassifier(

                                           silent = False,

                                           n_jobs=2,

                                           learning_rate=learning_rate,

                                           max_depth=int(round(max_depth)),

                                           n_estimators=int(round(n_estimators))

                                           )

                             )



    # 12.2 Now fit the pipeline and evaluate

    """Perform 10 fold cross validation of a model"""

    cv_result = cross_val_score(estimator = pipe_xg1,

                                X = train_set,

                                y = train_labels,

                                cv = 10,

                                n_jobs = -1,

                                scoring = scorer

                                ).mean()             # take the mean/max of all results





    # 12.3 Finally return maximum/average value of result

    return cv_result



#    return cv_result, pipe_xg1

# 12 This is the main workhorse

#      Instantiate BayesianOptimization() object

#

xgBO = BayesianOptimization(

                             xg_eval,     # Function to evaluate performance.

                             para_set     # Parameter set from where parameters will be selected

                             )

# 13. Gaussian process parameters

#     Modulate intelligence of Bayesian Optimization process

gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian

                                 # process.



# 14. Fit/train (so-to-say) the BayesianOptimization() object

#     Start optimization. 25minutes

#     Our objective is to maximize performance (results)

start = time.time()

xgBO.maximize(init_points=10,    # Number of randomly chosen points to

                                 # sample the target function before

                                 #  fitting the gaussian Process (gp)

                                 #  or gaussian graph

               n_iter=15,        # Total number of times the

               #acq="ucb",       # ucb: upper confidence bound

                                 #   process is to be repeated

                                 # ei: Expected improvement

               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration

              **gp_params

               )

end = time.time()

(end-start)/60

# 15. Get values of parameters that maximise the objective

#xgBO.res

type(xgBO.res) #If type is list then call max directly
#xgBO.res['max']

xgBO.max
xgBO.max['params']
xgBO.max['params']['learning_rate']

xgBO.max['params']['n_estimators']

xgBO.max['params']['max_depth']

xgBO.max['params']['n_components']
cv_score_xgBO = xg_eval(

    xgBO.max['params']['learning_rate'],

    xgBO.max['params']['n_estimators'],

    xgBO.max['params']['max_depth'],

    xgBO.max['params']['n_components']

)
model_results = model_results.append(pd.DataFrame({

    'model': 'XGBClassifier', 

    'cv_mean': cv_score_xgBO.mean(), 

    'cv_std': cv_score_xgBO.std()}, 

    index = [0]),

                                     ignore_index = True)
model_results
cv_score_xgBO.mean()
cv_score_xgBO.std()
pipe_xg1 = make_pipeline (ss(),

                          PCA(n_components=int(round(xgBO.max['params']['n_components']))),

                          XGBClassifier(

                              silent = False,

                              n_jobs=-1,

                              learning_rate=xgBO.max['params']['learning_rate'],

                              max_depth=int(round(xgBO.max['params']['max_depth'])),

                              n_estimators=int(round(xgBO.max['params']['n_estimators']))

                          )

                         )

pipe_xg1.fit(train_set, train_labels)
test_set.info()
predictions = pipe_xg1.predict(test_set)

#predictions = [round(value) for value in test_labels]
predictions.size
predictions
submission_base.info()
test_ids = list(final.loc[final['Target'].isnull(), 'idhogar'])

predictions = pd.DataFrame({'idhogar': test_ids,

                               'Target': predictions})



# Make a submission dataframe

submission = submission_base.merge(predictions, 

                                   on = 'idhogar',

                                   how = 'left').drop(columns = ['idhogar'])

    

# Fill in households missing a head

submission['Target'] = submission['Target'].fillna(4).astype(np.int8)

submission.to_csv('Anirudh_submission.csv', index = False)