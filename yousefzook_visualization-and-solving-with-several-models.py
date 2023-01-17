# import pandas
import pandas as pd

# load train and test as dataframes
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
# import display function from ipython
from IPython.display import display, HTML

# display the first rows of each dataset
display(train_df.head())
print("train shape: {}".format(train_df.shape))
display(test_df.head())
print("test shape: {}".format(test_df.shape))
train_df = train_df.drop( columns = ['Descript', 'Resolution'] )
train_df.head()
# lets see the statistics summary of locations
lons = train_df['X'] # longitudes 
lats = train_df['Y'] # latitudes

print ("Longitudes summary:")
print (lons.describe())
print ("\nLatitudes summary:")
print (lats.describe())
import matplotlib.pyplot as plt


plt.scatter(lons, lats)
plt.show()
# eliminate rows with latitudes out of San Francisco range
train_df = train_df.drop(train_df[(train_df['Y'] > 37.84) | (train_df['Y'] < 37.7)].index)
# eliminate rows with longitudes out of San Francisco range
train_df = train_df.drop(train_df[((train_df['X'] > -122.32) | (train_df['X'] < -122.52))].index)
train_df.describe()
new_lons = train_df['X'] # longitudes 
new_lats = train_df['Y'] # latitudes

# scatter plot for lons vs lats
plt.scatter(new_lons, new_lats)
plt.xlabel('lons')
plt.ylabel('lats')
plt.show()

# histogram plot for lons and lats
plt.hist(new_lons)
plt.xlabel('lons')
plt.ylabel('ocurrance number')
plt.show()
plt.hist(new_lats)
plt.xlabel('lats')
plt.ylabel('ocurrance number')
plt.show()
from collections import Counter

def printCategoriesOccurrence():
    
    categories = train_df['Category']
    # count the number of occurances for each category
    occurances = Counter(categories)
    sorted_occ = sorted(occurances.items(), key=lambda pair: pair[1], reverse=True)
    for key, value in sorted_occ:
        print (key, value)
    return sorted_occ
        
sorted_occ = printCategoriesOccurrence()
from mpl_toolkits.basemap import Basemap
import numpy as np
# minimum and maximum longitude and latitude for map zooming
lon_min = min(new_lons) 
lon_max = max(new_lons) 
lat_min = min(new_lats) 
lat_max = max(new_lats) 

fig = plt.figure(figsize=(24,12)) # to make plot bigger
fig.add_subplot(111, frame_on=False)

# Here, we add some padding with 0.01 to the map width and height
map = Basemap(
    llcrnrlon=lon_min-0.01,
    llcrnrlat=lat_min-0.01,
    urcrnrlon=lon_max+0.01,
    urcrnrlat=lat_max+0.01
)

parallels = np.arange(37,38,0.02)
meridians = np.arange(-122.6,-122.3,.02)
map.drawcounties()
map.drawparallels(parallels,labels=[False,True,True,False])
map.drawmeridians(meridians,labels=[True,False,False,True])
map.scatter(new_lons, new_lats)

plt.show()
data = train_df.groupby(['X', 'Y']).size().reset_index(name='occurances')
data = data[data.occurances < 500]

fig = plt.figure(figsize=(24,12)) # to make plot bigger
fig.add_subplot(111, frame_on=False)

# Here, we add some padding with 0.01 to the map width and height
map = Basemap(
    llcrnrlon=lon_min-0.01,
    llcrnrlat=lat_min-0.01,
    urcrnrlon=lon_max+0.01,
    urcrnrlat=lat_max+0.01
)

parallels = np.arange(37,38,0.02)
meridians = np.arange(-122.6,-122.3,.02)
map.drawcounties()
map.drawparallels(parallels,labels=[False,True,True,False])
map.drawmeridians(meridians,labels=[True,False,False,True])
# map.scatter(new_lons, new_lats)
x = data.X.as_matrix()
y = data.Y.as_matrix()
z = data.occurances.as_matrix()
mymap= map.contour(x, y, z, tri=True)
map.colorbar(mymap,location='bottom',pad="5%")
plt.title("size="+str(15))
plt.show()
from matplotlib import gridspec

plt.subplots(figsize=(24, 78))
i = 0 # subplot number

grid = gridspec.GridSpec(13,3)
for pair in sorted_occ[:]:
    ax = plt.subplot(grid[i])
    map = Basemap(
        llcrnrlon=lon_min-0.01,
        llcrnrlat=lat_min-0.01,
        urcrnrlon=lon_max+0.01,
        urcrnrlat=lat_max+0.01
    )
    parallels = np.arange(37,38,0.02)
    meridians = np.arange(-122.6,-122.3,.02)
    map.drawcounties()
    map.drawparallels(parallels,labels=[False,True,True,False])
    map.drawmeridians(meridians,labels=[True,False,False,True])
    category_data = train_df[train_df['Category'] == pair[0]]
    
    lons = category_data['X']
    lats = category_data['Y']
    map.scatter(lons, lats)
    plt.title(pair[0]) # pair[0] = category name
    i+=1

plt.show()
    
import re

    
plt.subplots(figsize=(12, 36))
i = 0 # subplot number

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
category = sorted_occ[0][0]
category_data = train_df[train_df['Category'] == category]

def getMonthData(month):
    regex = "((19[0-9][0-9])|(20[0-9][0-9])-" + month + "-[0-9]* (\w|:)*)"
    columns = ['Dates', 'Category', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']
    result = pd.DataFrame(columns=columns)
    for index, row in category_data.iterrows():
        if re.match(regex,row['Dates']):
            row = row.transpose()
            result = result.append(row)
            category_data.drop(index)
    return result

grid = gridspec.GridSpec(6,2)
for month in months:
    ax = plt.subplot(grid[i])
    map = Basemap(
        llcrnrlon=lon_min-0.01,
        llcrnrlat=lat_min-0.01,
        urcrnrlon=lon_max+0.01,
        urcrnrlat=lat_max+0.01
    )
    parallels = np.arange(37,38,0.02)
    meridians = np.arange(-122.6,-122.3,.02)
    map.drawcounties()
    map.drawparallels(parallels,labels=[False,True,True,False])
    map.drawmeridians(meridians,labels=[True,False,False,True])
    month_data = getMonthData(month)
    lons = month_data['X']
    lats = month_data['Y']
    map.scatter(lons, lats)
    plt.title(month + "  size=" + str(len(month_data)))
    i+=1

plt.show()

plt.subplots(figsize=(12, 36))
i = 0 # subplot number

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
category = sorted_occ[0][0]
category_data = train_df[train_df['Category'] == category]

grid = gridspec.GridSpec(7,1)
for day in days:
    ax = plt.subplot(grid[i])
    map = Basemap(
        llcrnrlon=lon_min-0.01,
        llcrnrlat=lat_min-0.01,
        urcrnrlon=lon_max+0.01,
        urcrnrlat=lat_max+0.01
    )
    parallels = np.arange(37,38,0.02)
    meridians = np.arange(-122.6,-122.3,.02)
    map.drawcounties()
    map.drawparallels(parallels,labels=[False,True,True,False])
    map.drawmeridians(meridians,labels=[True,False,False,True])
    day_data = category_data[category_data['DayOfWeek'] == day]
    lons = day_data['X']
    lats = day_data['Y']
    map.scatter(lons, lats)
    plt.title(day + "  size=" + str(len(day_data)))
    i+=1

plt.show()
import math
# if category size < 1000, duplicate it to be = 1000
for key,value in sorted_occ:
    if value<1000:
        
        temp = train_df[train_df['Category'] == key]
        train_df = train_df.append([temp]*int(math.ceil((1000-value)/float(value))), ignore_index=True)

sorted_occ = printCategoriesOccurrence()
# spliting train data into target and other features
target = train_df['Category']
data = train_df.drop(columns=['Category'])
features = ['Dates', 'DayOfWeek', 'PdDistrict', 'Address']
for feature in features:
    print ("feature: {}    unique_size: {}".format(feature ,len(data[feature].unique())))
# convert given list of dates it will trim seconds, minutes and return result
def trimMinAndSecFromDates(dates):
    result = []
    for date in dates:
        result.append(date[:-6])
    return result

# trim minutes and seconds from dates
data['Dates'] = trimMinAndSecFromDates(data['Dates'])

# encode Dates using label encoding
data['Dates'] = data['Dates'].astype('category')
data['Dates_int'] = data['Dates'].cat.codes

# encode Address using label encoding
data['Address'] = data['Address'].astype('category')
data['Address_int'] = data['Address'].cat.codes

data.head()
data.drop(columns=['Dates', 'Address'], inplace=True)
data.head()
# get dummies for each feature
DayOfWeek_dummies = pd.get_dummies(data['DayOfWeek'])
PdDistrict_dummies = pd.get_dummies(data['PdDistrict'])

# join dummies to the original dataframe
data = data.join(DayOfWeek_dummies)
data = data.join(PdDistrict_dummies)

data.head()
data.drop(columns=['DayOfWeek', 'PdDistrict'], inplace=True)
print ("data size =",len(data))
data.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =  train_test_split(data, target, test_size=0.2, random_state=0, stratify=target)
print ("train size: {}, test size: {}".format(X_train.shape[0], X_test.shape[0]))
from sklearn.metrics import log_loss, fbeta_score
from time import time

# train function takes learner, the train data and target
def train_test_pipeline(learner, X_train, y_train, X_test, y_test):
    
    results = {}
    
    # training learner
    start = time()
    learner.fit(X_train, y_train)
    end = time()
    results['train_time'] = end - start
    
     # remove missed classes after fitting for logloss
    for category in list(set(target) - set(learner.classes_)):
        X_train = X_train.drop(y_train[y_train == category].index)
        y_train = y_train[y_train != category]
        X_test = X_test.drop(y_test[y_test == category].index)
        y_test = y_test[y_test != category]
    # predict samples in training set
    predictions = learner.predict(X_train)
    predictions_proba = learner.predict_proba(X_train)
    
    # calculate fbeta and log loss
    results['fscore'] = fbeta_score(y_train, predictions, beta=.5, average='micro')
    results['logloss'] = log_loss(y_train, predictions_proba)
    
    # predict testing samples and time of prediction
    start = time()
    predictions = learner.predict(X_test)
    predictions_proba = learner.predict_proba(X_test)
    end = time()
    results['test_time'] = end - start
    
    # calculate fbeta and log loss for testing set
    results['fscore_test'] = fbeta_score(y_test, predictions, beta=.5, average='micro')
    results['logloss_test'] = log_loss(y_test, predictions_proba)
    
    
    print ("{} trained".format(learner.__class__.__name__))
    
    return results

# do train_test_pipeline then visualize resutls for given models on first n samples and test on first m
# returns predictions proba for last model in given list - this will be used for 1 model only -
def train_test_models(models, names=None, n=len(y_train), m=len(y_test)):
    results = {}
    i = 0
    for model in models:
        if not names:
            model_name = model.__class__.__name__
        else:
            model_name = names[i]
            i += 1
        results[model_name] = train_test_pipeline(model, X_train[:n], y_train[:n], X_test[:m], y_test[:m])

    # print results
    for model in results:
        model_res = results[model]
        print ("model: {}".format(model))
        print ("fscore:\t\t{}\nlogloss:\t{}\ntrain time:\t{}".format(model_res['fscore'], model_res['logloss'], model_res['train_time']))
        print ("fscore_test:\t\t{}\nlogloss_test:\t{}\ntest time:\t{}".format(model_res['fscore_test'], model_res['logloss_test'], model_res['test_time']))

    # visualize the results    
    visualize(results, random_results)
import matplotlib.pyplot as plt
def visualize(results, random_results):
    bar_width = 0.3
    fig, ax = plt.subplots(6,1,figsize = (12,32))
    for j, metric in enumerate(['train_time', 'fscore', 'logloss', 'test_time', 'fscore_test', 'logloss_test']):
        ax[j].set_xlabel("Learners")
        ax[j].set_ylabel(metric)
        ax[j].set_title(metric)
        for k, learner in enumerate(results.keys()):
            ax[j].bar(learner, results[learner][metric], width=bar_width)
    
    # add horizontal line for random model results
    ax[0].axhline(y=random_results['train_time'], linestyle='dashed')
    ax[1].axhline(y=random_results['fscore'], linestyle='dashed')
    ax[2].axhline(y=random_results['logloss'], linestyle='dashed')  
    ax[3].axhline(y=random_results['test_time'], linestyle='dashed')
    ax[4].axhline(y=random_results['fscore_test'], linestyle='dashed')
    ax[5].axhline(y=random_results['logloss_test'], linestyle='dashed')        
import random
class random_model:
 
    def __init__(self, categories):
        self.categories = categories
        self.classes_ = categories

    # always return a random value from categories
    def __getRandomValue(self): return random.choice(self.categories) 
    
    # no need for fit here
    def fit(self, X_train, y_train): pass
    
    def predict(self, X):
        result = [[] for i in range(len(X))]
        for j in range(len(X)):
            result[j] = self.__getRandomValue()
        return result
        
    def predict_proba(self, X):
        result = [[] for i in range(len(X))]
        for j in range(len(X)):
            row = [0.0]*len(self.categories)
            prediction = self.__getRandomValue()
            for i in range(len(self.categories)): 
                if(self.categories[i] == prediction):
                    row[i] = 1.0
                    break
            result[j] = row
        return result
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier

# initializing models
model_KNN = KNeighborsClassifier(n_jobs=-1, weights='distance')
model_tree = DecisionTreeClassifier(class_weight='balanced')
model_extraTrees = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced')
model_NN = MLPClassifier(learning_rate='invscaling', shuffle=True)
model_SVC = SVC(probability=True, class_weight='balanced') # One-to-One
model_XGB = XGBClassifier(one_drop=1)
# get random model results
model_random = random_model(categories=target.unique())
random_results = train_test_pipeline(model_random, X_train, y_train, X_test, y_test)
models = [model_KNN, model_tree, model_extraTrees, model_NN, model_SVC, model_XGB]
train_test_models(models, n=10000, m=2000)
# training XGBClassifier on all samples
models = [model_XGB]
train_test_models(models)
# training other models on all data
models = [model_KNN, model_tree, model_NN]
train_test_models(models)
model_NN_tuned1 = MLPClassifier(learning_rate='invscaling', shuffle=True, learning_rate_init=0.001,
                               hidden_layer_sizes=100, solver='adam')
model_NN_tuned2 = MLPClassifier(learning_rate='adaptive', shuffle=True, learning_rate_init=0.001,
                               hidden_layer_sizes=100, solver='adam')
model_NN_tuned3 = MLPClassifier(learning_rate='adaptive', shuffle=True, learning_rate_init=0.001,
                               hidden_layer_sizes=50, solver='adam')
model_NN_tuned4 = MLPClassifier(learning_rate='adaptive', shuffle=True, learning_rate_init=0.0001,
                               hidden_layer_sizes=200, solver='adam')
model_NN_tuned5 = MLPClassifier(learning_rate='adaptive', shuffle=True, learning_rate_init=0.001,
                               hidden_layer_sizes=100, solver='lbfgs')
model_NN_tuned6 = MLPClassifier(learning_rate='adaptive', shuffle=True, learning_rate_init=0.001,
                               hidden_layer_sizes=100, solver='sgd')

models = [model_NN_tuned1, model_NN_tuned2,model_NN_tuned3,model_NN_tuned4,model_NN_tuned5,model_NN_tuned6]
names = ["model 1", "model 2", "model 3", "model 4", "model 5", "model 6"]
train_test_models(models, names=names)
model_NN_tuned = MLPClassifier(learning_rate='adaptive', shuffle=True, epsilon=1e-8, activation='relu',
                               hidden_layer_sizes=100, solver='adam', verbose=True)

models = [model_NN_tuned]
train_test_models(models)
test_df.head()
# encode Dates and Address
test_df['Dates'] = trimMinAndSecFromDates(test_df['Dates'])
test_df['Dates'] = test_df['Dates'].astype('category')
test_df['Dates_int'] = test_df['Dates'].cat.codes
test_df['Address'] = test_df['Address'].astype('category')
test_df['Address_int'] = test_df['Address'].cat.codes
test_df.drop(columns=['Dates', 'Address'], inplace=True)

# encode DayOfWeek and PdDistrict
DayOfWeek_dummies = pd.get_dummies(test_df['DayOfWeek'])
PdDistrict_dummies = pd.get_dummies(test_df['PdDistrict'])
test_df = test_df.join(DayOfWeek_dummies)
test_df = test_df.join(PdDistrict_dummies)
test_df.drop(columns=['DayOfWeek', 'PdDistrict'], inplace=True)

# drop id column
test_df.drop(columns=['Id'], inplace=True)

test_df.head()
results = model_NN_tuned.predict_proba(test_df)
results.shape
output_df = pd.DataFrame(data=results, columns=model_NN_tuned.classes_)
output_df.head()
output_df.index.name = 'Id'
output_df.to_csv("output.csv")