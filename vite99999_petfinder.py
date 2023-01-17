import glob

import json

import pandas as pd

import numpy as np

import scipy as sp

from functools import partial

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
kaggle_path = True
if kaggle_path:

    train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

    test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

    colors = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv', index_col=0)

    breeds = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')

else:

    train = pd.read_csv('train.csv')

    test = pd.read_csv('test.csv')

    colors = pd.read_csv('color_labels.csv', index_col=0)

    breeds = pd.read_csv('breed_labels.csv')



df = pd.concat([train, test], ignore_index=True, sort=False)
breeds['type_breed'] = breeds['Type']*1000+breeds['BreedID']

breeds = breeds['BreedName'].set_axis(breeds['type_breed'], inplace=False)
df.columns
# Processing breed columns

df['Type_Breed1'] = df['Type']*1000+df['Breed1']

df['Type_Breed2'] = df['Type']*1000+df['Breed2']

df['Breed'] = df['Type_Breed1'].apply(str) + df['Type_Breed2'].apply(str)

# For each breed combination, replace name with group count and mean target

df['Breed_count'] = df['Breed'].map(df['Breed'].value_counts())

#df['Breed_mean'] = df['Breed'].map(df.groupby('Breed')['AdoptionSpeed'].mean().fillna(df['AdoptionSpeed'].mean()))

# Create column for pure-breed animals

df['PureBreed'] = (df['Breed1']*df['Breed2']).map(lambda x: 1 if x==0 else 0)

df.loc[df['Breed1']==307, 'PureBreed'] = 0 # 307=Mixed

df.loc[df['Breed2']==307, 'PureBreed'] = 0 # 307=Mixed
# Convert color variables into sparse matrices

# probably not the best way but it works

for col in ['Color1', 'Color2', 'Color3']:

    df[col] = df[col].map(colors.to_dict()['ColorName'])

for color in colors['ColorName'].tolist():

    is_color = lambda x: x==color

    df[color] = df.apply(lambda x: is_color(x['Color1']) or is_color(x['Color2']) or is_color(x['Color3']), axis=1)
df['Type'] = df['Type'].map(lambda x: x-1) # 0=dog, 1=cat

df['Nameless'] = df['Name'].isna() # create column for animals with no name

df=pd.concat([df, pd.get_dummies(df['Gender'], prefix='Gender_')], axis=1)

df.loc[df['Name']=='No Name', 'Nameless'] = True # 73 animals named "No Name"

df['Has_fee'] = df['Fee'].map(lambda x: 1 if x>0 else 0)



df['Rescuer_freq'] = df['RescuerID'].map(df['RescuerID'].value_counts())

#df['Rescuer_mean'] = df['RescuerID'].map(df.groupby('RescuerID')['AdoptionSpeed'].mean().fillna(df['AdoptionSpeed'].mean()))

df['State_freq'] = df['State'].map(df['State'].value_counts())

#df['State_mean'] = df['State'].map(df.groupby('State')['AdoptionSpeed'].mean().fillna(df['AdoptionSpeed'].mean()))



for col in ['Vaccinated', 'Dewormed', 'Sterilized']:

    df[col] = df[col].map(lambda x: 1 if x==1 else 0)
for col in df.columns:

    if df[col].dtype == bool:

        df[col] = df[col].map(lambda x: 1 if x else 0)
df = df.drop(['Name', 'Breed1', 'Breed2', 'Gender', 'Gender__3',

              'Color1', 'Color2', 'Color3', 'Type_Breed1', 'Type_Breed2', 'Breed',

              'State', 'RescuerID', 'Description'], axis=1)
df.columns
if kaggle_path:

    train_sentiment_data = glob.glob('../input/petfinder-adoption-prediction/train_sentiment/*.json')

    test_sentiment_data = glob.glob('../input/petfinder-adoption-prediction/test_sentiment/*.json')

else:

    train_sentiment_data = glob.glob('train_sentiment/*.json')

    test_sentiment_data = glob.glob('test_sentiment/*.json')

print (len(train_sentiment_data), len(test_sentiment_data))
def load_sentiment(file):

    with open(file, 'r', encoding='utf-8') as f:

        j = json.load(f)

    magnitudes = np.array([x['magnitude'] for x in [y['sentiment'] for y in j['sentences']]])

    scores = np.array([x['score'] for x in [y['sentiment'] for y in j['sentences']]])

    text_mag_sum = magnitudes.sum()

    text_mag_avg = magnitudes.mean()

    text_mag_var = magnitudes.var()

    text_score_sum = scores.sum()

    text_score_avg = scores.mean()

    text_score_var = scores.var()

    doc_mag = j['documentSentiment']['magnitude']

    doc_score = j['documentSentiment']['score']

    return (doc_mag, doc_score, text_mag_sum, text_mag_avg, text_mag_var, text_score_sum, text_score_avg, text_score_var)
train_sentiment = {}

for file in train_sentiment_data:

    r = load_sentiment(file)

    petid = file.split('train_sentiment/')[1].split('.')[0]

    train_sentiment[petid] = r

    

test_sentiment = {}

for file in test_sentiment_data:

    r = load_sentiment(file)

    petid = file.split('test_sentiment/')[1].split('.')[0]

    test_sentiment[petid] = r
sentiment = pd.concat([pd.DataFrame(train_sentiment).T, pd.DataFrame(test_sentiment).T])

sentiment = sentiment.reset_index()

sentiment.columns=['PetID', 'doc_mag', 'doc_score', 'text_mag_sum', 'text_mag_avg', 

                   'text_mag_var', 'text_score_sum', 'text_score_avg', 'text_score_var']
df = df.merge(sentiment, how='left', on='PetID')

for col in list(sentiment.columns.drop('PetID')):

    df[col] = df[col].fillna(df[col].mean())
if kaggle_path:

    train_metadata_paths = glob.glob('../input/petfinder-adoption-prediction/train_metadata/*-1.json')

    test_metadata_paths = glob.glob('../input/petfinder-adoption-prediction/test_metadata/*-1.json')

else:

    train_metadata_paths = glob.glob('train_metadata/*-1.json')

    test_metadata_paths = glob.glob('test_metadata/*-1.json')

print (len(train_metadata_paths), len(test_metadata_paths))
def extract_colors(d):

    try:

        red = d['red']

    except KeyError:

        red = 0

    try:

        blue = d['blue']

    except KeyError:

        blue = 0

    try:

        green = d['green']

    except KeyError:

        green = 0

    return red, blue, green



def load_metadata(file):

    with open(file, 'r', encoding='utf-8') as f:

        j = json.load(f)

    try: 

        annotations = j['labelAnnotations']

        #label_descs = np.array([x['description'] for x in annotations])

        label_scores_mean = np.array([x['score'] for x in annotations]).mean()

        label_topicality_mean = np.array([x['topicality'] for x in annotations]).mean()

    except KeyError:

        label_scores_mean = np.nan

        label_topicality_mean = np.nan

    colors = j['imagePropertiesAnnotation']['dominantColors']['colors']

    color_rgbs = np.array([x['color'] for x in colors]) 

    reds = []

    blues = []

    greens = []

    for d in color_rgbs:

        red, blue, green = extract_colors(d)

        reds.append(red)

        blues.append(blue)

        greens.append(green)

    color_red_mean = np.array(reds).mean()

    color_blue_mean = np.array(blues).mean()

    color_green_mean = np.array(greens).mean()

    color_score_mean = np.array([x['score'] for x in colors]).mean()

    color_fraction_mean = np.array([x['pixelFraction'] for x in colors]).mean()

    crops = j['cropHintsAnnotation']['cropHints']

    crop_confidence_mean = np.array([x['confidence'] for x in crops]).mean()

    return (label_scores_mean, label_topicality_mean, 

            color_red_mean, color_blue_mean, color_green_mean,

            color_score_mean, color_fraction_mean,

            crop_confidence_mean)
train_metadata = {}

for file in train_metadata_paths:

    r = load_metadata(file)

    petid = file.split('train_metadata/')[1].split('-1.')[0]

    train_metadata[petid] = r

    

test_metadata = {}

for file in test_metadata_paths:

    r = load_metadata(file)

    petid = file.split('test_metadata/')[1].split('-1.')[0]

    test_metadata[petid] = r
metadata = pd.concat([pd.DataFrame(train_metadata).T, pd.DataFrame(test_metadata).T])

metadata = metadata.reset_index()

metadata.columns=['PetID', 'label_scores_mean', 'label_topicality_mean', 'color_red_mean', 'color_blue_mean', 'color_green_mean', 

                  'color_score_mean', 'color_fraction_mean', 'crop_confidence_mean']
df = df.merge(metadata, how='left', on='PetID')

for col in list(metadata.columns.drop('PetID')):

    df[col] = df[col].fillna(df[col].mean())
train_img_features = pd.read_csv('../input/extracted-image-features-petfinder/train_img_features.csv')

test_img_features = pd.read_csv('../input/extracted-image-features-petfinder/test_img_features.csv')

img_features = pd.concat([train_img_features, test_img_features], ignore_index=True)

new_cols = ['image_'+str(i) for i in range(256)]

new_cols.insert(0, 'PetID')

img_features.columns = new_cols
from sklearn.decomposition import PCA

img_X = img_features.drop('PetID', axis=1).values

n_components = 64

pca_model = PCA(n_components).fit(img_X)

print(pca_model.explained_variance_ratio_.sum())  

img_X_new = pca_model.transform(img_X)
img_features_new = pd.DataFrame(img_X_new)

img_features_new.columns = ['image_'+str(i) for i in range(n_components)]

img_features_new['PetID'] = img_features['PetID']
df = df.merge(img_features_new, how='left', on='PetID')

df.head()
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import cohen_kappa_score

import xgboost as xgb
class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0

    

    def _kappa_loss(self, coef, X, y):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return -cohen_kappa_score(y, preds, weights='quadratic')

    

    def fit(self, X, y):

        loss_partial = partial(self._kappa_loss, X = X, y = y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    

    def predict(self, X, coef):

        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])

        return preds

    

    def coefficients(self):

        return self.coef_['x']
params_xgb1 = {'eval_metric': 'rmse', 

               'mex_depth': 6,

               'seed': 0, 

               'eta': 0.01, 

               'gamma': 0.03,

               'subsample': 0.8, 

               'colsample_bytree': 0.85, 

               'colsample_bylevel': 0.85, 

               'silent': 1}
def run_xgb(params, X_train, X_test):

    n_splits = 10

    verbose_eval = 1000

    num_rounds = 60000

    early_stop = 500



    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))

    oof_test = np.zeros((X_test.shape[0], n_splits))



    i = 0



    for train_idx, valid_idx in kf.split(X_train, X_train['AdoptionSpeed'].values):



        X_tr = X_train.iloc[train_idx, :]

        X_val = X_train.iloc[valid_idx, :]

        y_tr = X_tr['AdoptionSpeed'].values

        X_tr = X_tr.drop(['AdoptionSpeed'], axis=1)

        y_val = X_val['AdoptionSpeed'].values

        X_val = X_val.drop(['AdoptionSpeed'], axis=1)

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)

        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        model = xgb.train(dtrain=d_train, num_boost_round=num_rounds, evals=watchlist,

                          early_stopping_rounds=early_stop, verbose_eval=verbose_eval, params=params)



        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)

        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)



        oof_train[valid_idx] = valid_pred

        oof_test[:, i] = test_pred



        i += 1

    return model, oof_train, oof_test
X_train_non_null = df[df['AdoptionSpeed'].notnull()].drop(['PetID'], axis=1)

X_test_non_null = df[df['AdoptionSpeed'].isnull()].drop(['PetID', 'AdoptionSpeed'], axis=1)

model, oof_train, oof_test = run_xgb(params_xgb1, X_train_non_null, X_test_non_null)
optR = OptimizedRounder()

optR.fit(oof_train, X_train_non_null['AdoptionSpeed'].values)

coefficients = optR.coefficients()

valid_pred = optR.predict(oof_train, coefficients)

qwk = cohen_kappa_score(X_train_non_null['AdoptionSpeed'], valid_pred, weights='quadratic')

print("QWK = %.4f" % qwk)
optR.coefficients()
test_predictions = optR.predict(oof_test.mean(axis=1), coefficients).astype(np.int8)
# Private: 0.38581

submission = pd.DataFrame({'PetID': test['PetID'].values, 'AdoptionSpeed': test_predictions})

submission.to_csv('submission.csv', index=False)

submission.head()
xgb.plot_importance(model, max_num_features=24, height=0.8)