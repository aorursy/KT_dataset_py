%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



sns.set() # seaborn attributet set default
train = pd.read_csv("../input/sf-crime/train.csv.zip")

test = pd.read_csv("../input/sf-crime/test.csv.zip")
train.head(10)
test.head(10)
print(len(train[train['Category'].isnull()]))

train = train[train['Category'].isnull() == False]

print(len(train[train['Category'].isnull()]))

print(len(test))
train.isnull().sum()
# categories :

train['Category'].unique()
if 'Descript' in train:

    train = train.drop(['Descript'], axis=1)

if 'Resolution' in train:

    train = train.drop(['Resolution'], axis=1)

train.head()
test.head()
def rebuild_datetime(df):

    df['Dates'] = pd.to_datetime(df['Dates'])

    df['Date'] = df['Dates'].dt.date

    df['Hour'] = df['Dates'].dt.hour

    df['Minute'] = df['Dates'].dt.minute

    df['DayOfWeek'] = df['Dates'].dt.weekday

    df['Month'] = df['Dates'].dt.month

    df['Year'] = df['Dates'].dt.year

    df['Block'] = df['Address'].str.contains('block', case=False)

    

    return df



train = rebuild_datetime(train)

test = rebuild_datetime(test)



# check wrong datetime exists.

print("wrong Dates(train):", len(train[train['Dates'].isnull()]))

print("wrong Dates(test):", len(test[test['Dates'].isnull()]))
from shapely.geometry import Point

import geopandas as gpd



def create_gdf(df):

    gdf = df.copy()

    gdf['Coordinates'] = list(zip(gdf.X, gdf.Y))

    gdf.Coordinates = gdf.Coordinates.apply(Point)

    gdf = gpd.GeoDataFrame(

        gdf, geometry='Coordinates', crs={'init': 'epsg:4326'})

    return gdf



train_gdf = create_gdf(train)



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world.plot(color='white', edgecolor='black')

train_gdf.plot(ax=ax, color='red')

plt.show()
wrongxycnt = lambda df : len(df[(df['X'] == -120.5) & (df['Y'] == 90.0)])

print(wrongxycnt(train))

print(wrongxycnt(test))



def fix_gps(df):

    cnt = 0

    d = df[(df['X'] == -120.5) & (df['Y'] == 90.0)]

    for idx, row in d.iterrows():

        district = row['PdDistrict']

        xys = df[df['PdDistrict'] == district][['X', 'Y']]

        #print("PdDistrict:", district)

        df.loc[idx, ['X']] = xys['X'].mean()

        df.loc[idx, ['Y']] = xys['Y'].mean()

        #print(df.loc[idx, ['X']].values[0], df.loc[idx, ['Y']].values[0])

        cnt = cnt + 1

    print('cnt', cnt)

    

def fix_gps_values():

    fix_gps(train)

    fix_gps(test)

    

def drop_wrong_gps(df):

    df = df.drop(df[df['X'] == -120.5].index)

    return df

    

fix_gps(train)

fix_gps(test)



print(wrongxycnt(train))

print(wrongxycnt(test))
print(train['Category'])
data = train.groupby('Category').count()

data = data['Dates'].sort_values(ascending=False)



plt.figure(figsize=(20, 12))

ax = sns.barplot((data.values / data.values.sum()) * 100,data.index)



plt.title('Count by Category', fontdict={'fontsize': 24})

plt.xlabel('Percentage')
fig, ax = plt.subplots(figsize=(24,16))

font = {'family' : 'normal',

        'weight' : 'bold',

        'size'   : 32}



import matplotlib

matplotlib.rc('font', **font)



fig.suptitle('Num. Incidents / Hour (Category)')

for category in train['Category'].unique():

    #print(category)

    data = train[train['Category'] == category].groupby('Hour')

    a = data['Hour'].count()

    ax.plot(a.index, data['Hour'].count(), label=category)

    

plt.legend()



plt.show()
fig, ax = plt.subplots(figsize=(24,16))



import matplotlib

matplotlib.rc('font', **font)



def plotbycol(df, col, title):

    fig.suptitle(title)

    for category in df['Category'].unique():

        #print(category)

        data = df[df['Category'] == category].groupby(col)

        a = data[col].count()

        ax.plot(a.index, data[col].count(), label=category)

    

    plt.legend()

    plt.show()

    

plotbycol(train, 'DayOfWeek', "Incidents by Week")
train['PdDistrict'].unique()
if 'Address' in train:

    train = train.drop(['Address'], axis=1)

    

if 'Address' in test:

    test = test.drop(['Address'], axis=1)
print(train.head())

print(test.head())
def show_incidents_count_by_year_graph(df):

    fig, ax = plt.subplots(1, 1, figsize=(24,16))



    data = df

    data['datetime'] = data['Date'].astype('datetime64')

    data['Year'] = data['datetime'].dt.year    



    for cat in df.Category.unique():

        curdata = data[data['Category'] == cat]

        counts = curdata.groupby('Year')

        a = counts.size()    

        x = list(a.index)

        y = list(a)    

        ax.plot(x, y, label=cat)



    plt.legend()

    plt.show()

    

show_incidents_count_by_year_graph(train)
if 'Dates' in train:

    train = train.drop(['Dates'], axis=1)

if 'Date' in train:

    train = train.drop(['Date'], axis=1)

    

if 'Dates' in test:

    test = test.drop(['Dates'], axis=1)

if 'Date' in test:

    test = test.drop(['Date'], axis=1)

    
train.head()
test.head()
if 'DoWN' in train:

    train = train.drop(['DoWN'], axis=1)



if 'DoWN' in test:

    test = test.drop(['DoWN'], axis=1)



if 'datetime' in train:

    train = train.drop(['datetime'], axis=1)

    

if 'Year' in train:

    train = train.drop(['Year'], axis=1)

    

if 'datetime' in test:

    test = test.drop(['datetime'], axis=1)

    

if 'Year' in test:

    test = test.drop(['Year'], axis=1)

    

train.head()
test.head()
test_ids = test['Id'].astype('int')
if "Id" in test:

    test = test.drop(['Id'], axis=1)
train_category = train['Category']
from sklearn.preprocessing import LabelEncoder



if 'Category' in train:

    train = train.drop(['Category'], axis=1)

    

train_X = train          



# names of categorical features.(need to pass LGBM model)

categoricals = ["PdDistrict"]



le_pdDistrict = LabelEncoder()

train_X['PdDistrict'] = le_pdDistrict.fit_transform(train_X['PdDistrict'])

test['PdDistrict'] = le_pdDistrict.transform(test['PdDistrict'])



le_category = LabelEncoder()

train_Y = le_category.fit_transform(train_category)

num_category = len(list(le_category.classes_))
len(list(le_category.classes_))
train_X
train_Y
print("train_category : ", train_category)
import eli5

from eli5.sklearn import PermutationImportance

from lightgbm import LGBMClassifier

import lightgbm as lgb

from sklearn.model_selection import train_test_split



def show_feature_importance(df_X, df_Y):

    params = {

        'n_estimators' : 3,

        'learning_rate' : 0.4,

        'max_delta_step' : 0.9,

        'min_data_in_leaf' : 21,

        'max_bin' : 465,

        'num_leaves' : 41,

    }



    _train_X, _val_X, _train_y, _val_y = train_test_split(df_X, df_Y)



    model = LGBMClassifier(objective='multiclass', num_class=num_category, n_estimators=200)

    model.set_params(**params)

    model.fit(_train_X, _train_y)



    perm = PermutationImportance(model).fit(_val_X, _val_y)

    display(eli5.show_weights(perm, feature_names=_val_X.columns.tolist(), include_styles=False))



    

show_feature_importance(train_X, train_Y)
import lightgbm as lgb

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn import linear_model



from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



import eli5

from eli5.sklearn import PermutationImportance



import warnings



from bayes_opt import BayesianOptimization



n_splits = 8



def get_param(learning_rate, max_delta_step, min_data_in_leaf, max_bin, num_leaves):

    params = {'n_estimators' : 400,

                'boosting_type' : 'gbdt',

                'objective' : 'multiclass',

                'max_delta_step': max_delta_step,

                'min_data_in_leaf': int(min_data_in_leaf),               

                'max_bin': int(max_bin),

                'num_leaves': int(num_leaves),

                'learning_rate' : learning_rate,

                'num_class' : num_category,

                'early_stopping_rounds': 5,

              }

    return params



def opt_test_func(learning_rate, max_delta_step, min_data_in_leaf, max_bin, num_leaves):

    

    params = get_param(learning_rate, max_delta_step, min_data_in_leaf, max_bin, num_leaves)

    print("params : ", params)

    acc, _ = train(params)

    return acc



def multiclass_logloss(predictions, labels, epsilon=1e-12):

    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    N = predictions.shape[0]

    loss = -np.sum(labels*np.log(predictions+1e-9))/N

    return loss



def train_cv(params):

    models = []

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=7)

    all_predictions = np.zeros((len(train_X), len(list(le_category.classes_))))



    for train_idx, test_idx in kfold.split(train_X, train_Y):

        X_train, y_train = train_X.loc[train_idx], train_Y[train_idx]

        X_valid, y_valid = train_X.loc[test_idx], train_Y[test_idx]

        

        #print(X_train.shape, y_train.shape)



        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categoricals)

        valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categoricals)

        model = lgb.train(params, train_set=train_data, num_boost_round=120, valid_sets=[valid_data], verbose_eval=5)        



        pred = model.predict(X_valid)

#         print("pred.shape :", pred.shape)

#         print("all_predictions.shape : ", all_predictions.shape)

        

        all_predictions[test_idx] = pred        

        models.append(model)

        

    labels_one_hot = np.eye(len(list(le_category.classes_)))[train_Y]

    loss = multiclass_logloss(all_predictions, labels_one_hot)

    print("validation multiclass logloss :", loss)

    

    return models, loss





def bo_eval_func(learning_rate, max_delta_step, min_data_in_leaf, max_bin, num_leaves):

    params = get_param(learning_rate, max_delta_step, min_data_in_leaf, max_bin, num_leaves)

    _, loss = train_cv(params)

    return -loss

    

    

def get_optimized_hyperparameters():

    """use this function for refining hyperparameters

    

    Returns:

        dictionary of hyperparameters

    """

    bo_params = {        

        'learning_rate' : (0.01, 0.4),

        'max_delta_step': (0.5, 2.5),

        'min_data_in_leaf': (15, 45),

        'max_bin': (200, 500),

        'num_leaves': (20, 50),

    }

    

    optimizer = BayesianOptimization(bo_eval_func, bo_params, random_state=1030)

    

    with warnings.catch_warnings():

        warnings.filterwarnings('ignore')

        init_points = 16

        n_iter = 16

        optimizer.maximize(init_points = init_points, n_iter = n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        return optimizer.max['params']





#params = get_optimized_hyperparameters()



params = {

    'learning_rate' : 0.4,

    'max_delta_step' : 0.9,

    'min_data_in_leaf' : 21,

    'max_bin' : 465,

    'num_leaves' : 41,

}



params = get_param(**params)

models, loss = train_cv(params)

print("train loss : ", loss)
train_X
test
def predict(models, test):

    preds = []

    for model in models:

        pred = model.predict(test)        

        preds.append(pred)



    predsCnt = len(preds)

    preds = np.array(preds)

    preds = np.sum(preds, axis=0) / predsCnt

    return preds



pred = predict(models, test)



submission = pd.DataFrame(pred, columns=le_category.inverse_transform(np.linspace(0, 38, 39, dtype='int')), index=test.index)

submission.to_csv('submission.csv', index_label='Id')