import numpy as np 
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from collections import Counter
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import json

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')
sub = pd.read_csv('../input/sample_submission.csv')
print('size of train data',train.shape)
print('size of test data',test.shape)
train.head()
test.head()
# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_train_data.head(20)
# checking missing data
total = test.isnull().sum().sort_values(ascending = False)
percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)
missing_test_data  = pd.concat([total, percent], axis=1, keys=['Total missing', 'Percent missing'])
missing_test_data.head(20)
temp = train['cuisine'].value_counts()
trace = go.Bar(
    y=temp.index[::-1],
    x=(temp / temp.sum() * 100)[::-1],
    orientation = 'h',
    marker=dict(
        color='blue',
    ),
)

layout = go.Layout(
    title = "Top cuisine",
    xaxis=dict(
        title='Recipe count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='Cuisine',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
n=6714 # total ingredients in train data
top = Counter([item for sublist in train.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
trace = go.Bar(
    y=temp.ingredient[::-1],
    x=temp.total_count[::-1],
    orientation = 'h',
    marker=dict(
        color='green',
    ),
)

layout = go.Layout(
    title = "Top ingredients",
    xaxis=dict(
        title='ingredient count',
        tickfont=dict(size=14,)),
    yaxis=dict(
        title='ingredient',
        titlefont=dict(size=16),
        tickfont=dict(
            size=14)),
    margin=dict(
    l=200,
),
    
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
def read_dataset(path):
	return json.load(open(path)) 
train = read_dataset('../input/train.json')
test = read_dataset('../input/test.json')

def generate_text(data):
	text_data = [" ".join(doc['ingredients']).lower() for doc in data]
	return text_data 

train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
    	x = tfidf.fit_transform(txt)
    else:
	    x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 
X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")
lb = LabelEncoder()
y = lb.fit_transform(target)

# Model Training 
classifier = SVC(C=100, # penalty parameter
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value
	 			 gamma=1, # kernel coefficient
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	      		 probability=False, # no need to enable probability estimates
	      		 cache_size=200, # 200 MB cache size
	      		 class_weight=None, # all classes are treated equally 
	      		 verbose=False, # print the logs 
	      		 max_iter=-1, # no limit, let it run
          		 decision_function_shape=None, # will use one vs rest explicitly 
          		 random_state=None)
model = OneVsRestClassifier(classifier, n_jobs=4)
model.fit(X, y)

# Predictions 
y_test = model.predict(X_test)
y_pred = lb.inverse_transform(y_test)

# Submission
test_id = [doc['id'] for doc in test]
sub = pd.DataFrame({'id': test_id, 'cuisine': y_pred}, columns=['id', 'cuisine'])
sub.to_csv('sub1.csv', index=False)