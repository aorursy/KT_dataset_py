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
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

import sqlite3
import sklearn, numpy
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

con = sqlite3.connect('../input/european-football/database.sqlite')

df = pd.read_sql_query("select * from football_data;", con)
#df = df.sample(frac=0.01)

print("Initial dataframe shape:", df.shape)

index_names = df[(df['FTR'] != 'H') & (df['FTR'] != 'A') & (df['FTR'] != 'D')].index

#df = df.sort_values('Datetime')

df.drop(index_names, inplace=True)

column_names = list(df.columns.values)
#df.dropna(inplace=True, thresh=int(0.8*df.shape[0]) , axis=1)
# The columns that contain significant number of null values
noneSet = []
noneSet = ['ABP', 'AFKC', 'AHW', 'AO', 'AT', 'Attendance', 'BSA', 'BSD', 'BSH', 
           'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 'BbAv>2.5', 'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD', 
           'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 'B365AH', 'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 
           'BbOU', 'GB<2.5', 'GB>2.5', 'GBA', 'GBAH', 'GBAHA', 'GBAHH', 'GBD', 'GBH', 'HBP', 'HFKC', 'HHW', 
           'HO', 'HT', 'LBAH', 'LBAHA', 'LBAHH', 'SBA', 'SBD', 'SBH', 'SJA', 'SJD', 'SJH', 'SOA', 'SOD', 'SOH',
           'SYA', 'SYD', 'SYH', 'PA', 'PD', 'PH', 'LBH', 'LBD', 'LBA']

allow_halftime = False

# The value of these columns is not known before the match
if not allow_halftime:
    results = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR']
else:
    # To allow half time statistics, use this
    results = ['FTHG', 'FTAG', 'FTR', 'HTR']

# The value of these columns are not known before the match as well
match_statistics = ['Attendance', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HHW', 'AHW', 'HC', 'AC', 'HF', 'AF', 'HFKC', 
 'AFKC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP']

X = df.drop(set(results + match_statistics + ['Datetime', 'Season', 'HT', 
                                              'AT']).intersection(column_names), axis=1)
print(X.shape)

# Desired result
y = df['FTR']
import datetime, time

# https://stackoverflow.com/questions/57330482/convert-data-frame-datatime-string-to-float-in-python-pandas
def convertDate(dateString):
    dateTime1 = datetime.datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S')
    #ignores time of the day
    return int(time.mktime(dateTime1.timetuple()))//86400 

def convertTime(timeString):
    hour, minute = map(int, timeString.split(':'))
    return hour*60+minute

def reciprocal(x):
    if x == 0:
        return 0
    return 1/float(x)

df['Date'] = df['Date'].apply(convertDate)
df['Time'] = df['Time'].apply(convertTime)

# Here are a list of betting odds field
# Taken from http://www.football-data.co.uk/notes.txt
betting_odds = ['B365H', 'B365D', 'B365A', 'BSH', 'BSD', 'BSA', 'BWH', 'BWD', 'BWA', 'GBH', 'GBD', 
'GBA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PH', 'PD', 'PA', 'SOH', 'SOD', 'SOA', 'SBH', 'SBD', 'SBA',
'SJH', 'SJD', 'SJA', 'SYH', 'SYD', 'SYA', 'VCH', 'VCD', 'VCA', 'WHH', 'WHD', 'WHA', 'Bb1X2', 'BbMxH', 'BbAvH', 
'BbMxD', 'BbAvD', 'BbMxA', 'BbAvA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'B365CH', 'B365CD', 'B365CA', 
'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA', 'VCCH', 'VCCD', 'VCCA', 'WHCH', 'WHCD', 'WHCA', 'MaxCH', 'MaxCD', 
'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA', 'PSH', 'PSD', 'PSA', 'PSCA', 'PSCD', 'PSCH']

num_goals_odds = ['BbMx>2.5', 'BbAv>2.5',
'BbMx<2.5', 'BbAv<2.5', 'GB>2.5', 'GB<2.5', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5', 
'Avg>2.5', 'Avg<2.5', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 
'MaxC<2.5', 'AvgC>2.5', 'AvgC<2.5']

asian_odds = ['AHh', 'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'GBAHH', 'GBAHA','GBAH','LBAHH',
'LBAHA','LBAH','B365AHH','B365AHA','B365AH','PAHH','PAHA','MaxAHH','MaxAHA','AvgAHH','AvgAHA', 
'B365CAHH','B365CAHA','PCAHH','PCAHA','MaxCAHH','MaxCAHA','AvgCAHH','AvgCAHA']

for i in betting_odds:
    df[i].fillna(2.85, inplace=True)

for i in num_goals_odds+asian_odds:
    df[i].fillna(1.90, inplace=True)

column_names = list(df.columns.values)

#Replace empty values with -1
for i in column_names:
    df[i].fillna(-1, inplace=True)


for col in betting_odds+num_goals_odds+asian_odds:
    try:
        df[col] = df[col].apply(reciprocal)
    except KeyError:
        continue
        #print(col, "KeyError")
    except:
        print(col)
        raise

        
def conv_BbAHh(i):
    try:
        return float(i)
    except:
        # print(i)
        a, b = map(float, i.split(','))
        # print(a, b)
        return (a+b)/2

try:
    df['BbAHh'] = df['BbAHh'].apply(conv_BbAHh)
except:
    pass

print("Final dataframe shape:", df.shape)
print(df.head(10))
X = df.drop(set(results + match_statistics + ['Datetime', 'Season', 'HT', 
                                              'AT']).intersection(column_names), axis=1)
print(X.shape)

# Desired result
y = df['FTR']

ce_binaryX = ce.BinaryEncoder(cols = ['HomeTeam', 'AwayTeam', 'Div', 'League', 'Country'])
ohe = ce.OneHotEncoder(cols = [])
X = ce_binaryX.fit_transform(X)
X = ohe.fit_transform(X)

label_encoder = LabelEncoder()
label_encoder.fit(['H', 'D', 'A'])
print(y[:10])
y = label_encoder.fit_transform(y)
print(y[:10])
'''
import matplotlib.pyplot as plt
columns = list(X.columns)
print(columns[0])
for i in columns:
    plt.scatter(X[i],y)
    plt.ylabel('Win')
    plt.title('Scatter plot of Win against ' + i)
    plt.show()
'''
seed = 153

# 80-20 split on training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from hmmlearn import hmm # Hidden Markovnikov model

scaler = StandardScaler()
pca = PCA()
pca10 = PCA(n_components=10)
pipeline = make_pipeline(scaler,pca)
pipeline10 = make_pipeline(scaler,pca10)

pipeline.fit(X_train)
pipeline10.fit(X_train)

# Plot the explained variances of the first 10 components
features = range(pca10.n_components_)
plt.bar(features, pca10.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
# shown above that only first 4 PCA features have significant variance
pca = PCA(n_components=4)
scaler = StandardScaler()
normalizer = Normalizer()
kmeans = KMeans(n_clusters=3)

kpipeline = make_pipeline(pca,kmeans)
spipeline = make_pipeline(pca,scaler,kmeans)
npipeline = make_pipeline(pca,normalizer,kmeans)

kmeans.fit(X_train)
kpipeline.fit(X_train)
spipeline.fit(X_train)
npipeline.fit(X_train)
pca = PCA(n_components=4)
scaler = StandardScaler()
normalizer = Normalizer()
#sc = SpectralClustering(3, assign_labels='discretize')

scpipeline = make_pipeline(pca,sc)
sscpipeline = make_pipeline(pca,scaler,sc)
nscpipeline = make_pipeline(pca,normalizer,sc)

scpipeline.fit(X_train)
sscpipeline.fit(X_train)
nscpipeline.fit(X_train)
pca = PCA(n_components=4)
normalizer = Normalizer()
gm = hmm.GaussianHMM(n_components=3)

ngmpipeline = make_pipeline(pca,normalizer,gm)

ngmpipeline.fit(X_train)
# The presence of PCA improves raw performance by reducing the number of computations for features with low variance

print("Outcome Legend (Unsupervised Learning)")
print("0: Away win")
print("1: Draw")
print("2: Home win")
print("Note: Supervised learning uses a different outcome legend\n")

kmeans.fit(X_train, y_train)
labels = kmeans.predict(X_test)
df1 = pd.DataFrame({'labels': labels, 'wins': y_test})
ct = pd.crosstab(df1['labels'],df1['wins'])
print('kmeans only')
print(ct)
print("Score:", kmeans.score(X_test, y_test))
print()

kpipeline.fit(X_train, y_train)
labels = kpipeline.predict(X_test)
df1 = pd.DataFrame({'labels': labels, 'wins': y_test})
ct = pd.crosstab(df1['labels'],df1['wins'])
print('kmeans with PCA')
print(ct)
print("Score:", kpipeline.score(X_test, y_test))
print()

spipeline.fit(X_train, y_train)
labels = spipeline.predict(X_test)
df1 = pd.DataFrame({'labels': labels, 'wins': y_test})
ct = pd.crosstab(df1['labels'],df1['wins'])
print('kmeans with PCA & scaler')
print(ct)
print("Score:", spipeline.score(X_test, y_test))
print()

npipeline.fit(X_train, y_train)
labels = npipeline.predict(X_test)
df1 = pd.DataFrame({'labels': labels, 'wins': y_test})
ct = pd.crosstab(df1['labels'],df1['wins'])
print('kmeans with PCA & normalizer')
print(ct)
print("Score:", npipeline.score(X_test, y_test))
print()
# F1 = 2 * (precision * recall) / (precision + recall)
# Using weighted average to account for label imbalance
from sklearn.metrics import f1_score, accuracy_score

kmeans.fit(X_train, y_train)
labels = kmeans.predict(X_test)
print('kmeans')
print('f1 score: '+ str(f1_score(y_test, labels, average='weighted')))
print('accuracy: '+ str(accuracy_score(y_test, labels)))
print()

kpipeline.fit(X_train, y_train)
labels = kpipeline.predict(X_test)
print('kmeans with PCA')
print('f1 score: '+ str(f1_score(y_test, labels, average='weighted')))
print('accuracy: '+ str(accuracy_score(y_test, labels)))
print()

spipeline.fit(X_train, y_train)
labels = spipeline.predict(X_test)
print('kmeans with PCA & scaler')
print('f1 score: '+ str(f1_score(y_test, labels, average='weighted')))
print('accuracy: '+ str(accuracy_score(y_test, labels)))
print()

npipeline.fit(X_train, y_train)
labels = npipeline.predict(X_test)
print('kmeans with PCA & normalizer')
print('f1 score: '+ str(f1_score(y_test, labels, average='weighted')))
print('accuracy: '+ str(accuracy_score(y_test, labels)))
print()

'''
#ngmpipeline.fit(X_train, y_train)
labels = ngmpipeline.fit_predict(X_test)
print('hmm with PCA & normalizer')
print('f1 score: '+ str(f1_score(y_test, labels, average='weighted')))
print('accuracy: '+ str(accuracy_score(y_test, labels)))
print()
'''

print('Hence shown that hmm with PCA & normalizer performs the worst')
from sklearn.ensemble import RandomForestClassifier

t1 = time.time()

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print("Accuracy:", rfc.score(X_test, y_test))

t2 = time.time()
print("Time taken:", t2-t1, "seconds")
from sklearn.ensemble import ExtraTreesClassifier

t1 = time.time()
etc = ExtraTreesClassifier()
etc.fit(X_train, y_train)

etc_pred = etc.predict(X_test)

print("Accuracy:", etc.score(X_test, y_test))
t2 = time.time()
print("Time taken:", t2-t1, "seconds")
from sklearn.linear_model import LogisticRegression

print("WARNING! This may take 10 minutes or so on Kaggle (2020)")

t1 = time.time()

for iters in range(10,201,10):
    lr = LogisticRegression(max_iter=iters)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    print("Number of iterations:",  iters, "of 200")
    #print(lr.score(X_test, y_test))

    t2 = time.time()
    print("Time taken:", t2-t1, "seconds")
print(lr.score(X_test, y_test))
print("Random Forest Classifier")
print(rfc.predict_proba(X_test)[:10])
print("Extra Trees Classifier")
print(etc.predict_proba(X_test)[:10])
print("Logistic Regression")
print(lr.predict_proba(X_test)[:10])
pred = {'RandomForestClassifier' : rfc_pred, 'ExtraTreesClassifier': etc_pred, 'LogisticRegression': lr_pred,
        'Actual': y_test}
res_df = pd.DataFrame(pred, columns = ['RandomForestClassifier', 'ExtraTreesClassifier', 'LogisticRegression',
                                       'Actual'])
print(res_df.head(10))
from sklearn.metrics import confusion_matrix
print("Outcome Legend")
print("0: Away win")
print("1: Draw")
print("2: Home win")
print()
print("Confusion matrix for Random Forest Classifier")
print(confusion_matrix(y_test, res_df['RandomForestClassifier']))
print("Confusion matrix for Extra Trees Classifier")
print(confusion_matrix(y_test, res_df['ExtraTreesClassifier']))
print("Confusion matrix for Logistic Regression")
print(confusion_matrix(y_test, res_df['LogisticRegression']))
from sklearn.ensemble import VotingClassifier
t1 = time.time()
eclf1 = VotingClassifier(estimators=[('1', rfc), ('2', etc), ('3', lr)], voting='hard')
eclf1.fit(X_train, y_train)
print("Accuracy:", eclf1.score(X_test, y_test))
t2 = time.time()
print("Time taken:", t2-t1, "seconds")
from sklearn.ensemble import VotingClassifier
t1 = time.time()
eclf2 = VotingClassifier(estimators=[('1', rfc), ('2', etc), ('3', lr)], voting='soft')
eclf2.fit(X_train, y_train)
print("Accuracy:", eclf2.score(X_test, y_test))
t2 = time.time()
print("Time taken:", t2-t1, "seconds")
print("Confusion matrix for Voting Classifier (hard voting)")
print(confusion_matrix(y_test, eclf1.predict(X_test)))
print("Confusion matrix for Voting Classifier (soft voting)")
print(confusion_matrix(y_test, eclf2.predict(X_test)))
import tensorflow as tf

print(X_train.shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(128, activation='tanh'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(3)
])

X_train_numpy = np.asarray(X_train)
y_train_numpy = np.asarray(y_train)
X_test_numpy = np.asarray(X_test)
y_test_numpy = np.asarray(y_test)

predictions = model(np.ndarray.astype(X_train_numpy[:1], np.float32)).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

pre_training_result = model.evaluate(np.ndarray.astype(X_test_numpy, np.float32), 
                                     np.ndarray.astype(y_test_numpy, np.float32), verbose=2)
print("Pre-training loss:", pre_training_result[0])
print("Pre-training accuracy:", pre_training_result[1])

model.fit(np.ndarray.astype(X_train_numpy, np.float32),
          np.ndarray.astype(y_train_numpy, np.float32), epochs=10, validation_split=0.2, batch_size=32)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = model(np.ndarray.astype(X_test_numpy, np.float32)).numpy()
forecast = np.argmax(predictions, axis=1)

post_training_result = model.evaluate(np.ndarray.astype(X_test_numpy, np.float32), 
                                      np.ndarray.astype(y_test_numpy, np.float32), verbose=2)
print("Post-training loss:", post_training_result[0])
print("Post-training accuracy:", post_training_result[1])
print()

print("Probability prediction")
print(tf.nn.softmax(predictions).numpy()[:10])
print("Result prediction")
print(forecast[:10])

print("Confusion matrix: ")
print(confusion_matrix(y_test, forecast))