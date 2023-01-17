# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import pickle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data  
regseason = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')
#/kaggle/input/march-madness-analytics-2020/WDataFiles_Stage2/WRegularSeasonDetailedR/esults.csv
postseason = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)
teams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Womens-Data/WDataFiles_Stage1/WTeamConferences.csv')

start = (teams.Season.values == 2003).argmax()
teams = teams.iloc[start:]

# set variables of necessary data
seasonIndex = games.columns.get_loc('Season')
WTeamIDIndex = games.columns.get_loc('WTeamID')
LTeamIDIndex = games.columns.get_loc('LTeamID')
WTeamMetrics = ['WScore', 'LScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM',
                'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
WTeamIndexes = []
LTeamMetrics = ['LScore', 'WScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM',
                'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
LTeamIndexes = []

# store index of necessary data within dataset
for i in range(0, len(WTeamMetrics)):
    index = games.columns.get_loc(WTeamMetrics[i])
    WTeamIndexes.append(index)

for i in range(0, len(LTeamMetrics)):
    index = games.columns.get_loc(LTeamMetrics[i])
    LTeamIndexes.append(index)
    
# -----------------------------
# DATASET SKELETON PREPARATION
# -----------------------------
cols = 5 + len(WTeamMetrics)
rows = teams.shape[0]

dataset = np.zeros(shape = (rows, cols))
dataset = dataset.astype(float)
teamDetails = teams.iloc[:, 0:2].values
dataset[:, 0:2] = teamDetails

rawData = games.iloc[:, :].values
index = games.columns.get_loc("WLoc")
for i in range(0, rawData.shape[0]):
    rawData[i][index] = 0
rawData = rawData.astype(float)


# -------------------------
# DATA PROCESSING
# -------------------------
for i in range(0, rawData.shape[0]):
    
    # output to console
    print("Game #: ", i)
    
    # get descriptive data of game
    thisWTeamID = rawData[i][WTeamIDIndex]
    thisLTeamID = rawData[i][LTeamIDIndex]
    thisSeason = rawData[i][seasonIndex]
    
    # initialize variables
    datasetWTeam = -1
    datasetLTeam = -1
    
    # locate rows of winning and losing teams
    for j in range(0, dataset.shape[0]):
        if(dataset[j][0] == thisSeason):
            if(dataset[j][1] == thisWTeamID):
                datasetWTeam = j
            elif(dataset[j][1] == thisLTeamID):
                datasetLTeam = j
    
    # add a win, a game played and the necessary data to winning team         
    dataset[datasetWTeam][2] += 1
    dataset[datasetWTeam][-1] += 1
    k = 0
    for j in WTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1
    
    # add a loss, a game played and the necessary data to losing team  
    dataset[datasetLTeam][3] += 1
    dataset[datasetLTeam][-1] += 1
    k = 0
    for j in LTeamIndexes:
        dataset[datasetWTeam][k + 4] += rawData[i][j]
        k += 1

# convert totals to averages for statistics       
for i in range(0, dataset.shape[0]):
    games = dataset[i][dataset.shape[1] - 1]
    for j in range(2, dataset.shape[1] - 1):
        if(games != 0): 
            val = float(dataset[i][j]) / float(games)
            val = round(val, 3)
            dataset[i][j] = val


# -------------------------
# DATA STORAGE
# -------------------------    
# save entire dataset to disk
np.savetxt("/kaggle/working/all_teamData.csv", dataset, delimiter = ",")

# output feedback
print()
print()
print("All Team Dataset creation complete.")
print()
print()

# locate index where 2019 data begins
found = -1
i = 0
while(found == -1):
    if(dataset[i][0] == 2019):
        found = i
    i += 1
   
# store subset dataset containing only data from 2019 
curr_year = dataset[found:, :]
np.savetxt("/kaggle/working/2020_teamData.csv", curr_year, delimiter = ",")

# output feedback
print("2020 Team Dataset creation complete.")
print()
print()

# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data 
teamdetails = pd.read_csv('/kaggle/working/all_teamData.csv', header = None)
rawTeamData = teamdetails.iloc[:, :].values

regseason = pd.read_csv('/kaggle/input/march-madness-analytics-2020/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv')
postseason = pd.read_csv('/kaggle/input/march-madness-analytics-2020/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv')
frames = [regseason, postseason]
games = pd.concat(frames)


# -----------------------------
# DATASET SKELETON PREPARATION
# -----------------------------
cols = 23
rows = games.shape[0]
dataset = np.zeros(shape = (rows, cols))
dataset = dataset.astype(float)


# -------------------------
# DATA PROCESSING
# -------------------------
for i in range(0, games.shape[0]):
    
    # output to console
    print("Game #: ", i)
    
    # get descriptive data of game
    season = games.iloc[i]['Season']
    WTeamID = games.iloc[i]['WTeamID']
    LTeamID = games.iloc[i]['LTeamID']
    
    # identify location of game (home, away or neutral)
    x = games.iloc[i]['WLoc']
    loc = 0
    if(x == 'N'):
        loc = 0
    elif(x == 'H'):
        loc = 1
    elif(x == 'A'):
        loc = -1
    
    # convert array to float
    values = np.array([i+1, season])
    values = values.astype(float)
          
    # add raw team data of winning team
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == WTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                WTeamData = rawTeamData[j, 2:-1]
        j += 1
        
    # add raw team data of losing team
    found = -1
    j = 0
    while(found == -1):
        if(rawTeamData[j][1] == LTeamID):
            if(rawTeamData[j][0] == season):
                found = j
                LTeamData = rawTeamData[j, 2:-1]
        j += 1
    
    # find difference between winning and losing team statistics
    winner = -1
    difference = np.subtract(WTeamData, LTeamData)
    
    # randomize for which will come first, and modify data accordingly
    x = random.uniform(0, 1)
    if(x > 0.5):
        winner = 1
        values = np.append(values, WTeamID)
        values = np.append(values, LTeamID)
        values = np.append(values, loc)
    else:
        winner = 0
        loc = loc * -1
        difference = difference * -1
        values = np.append(values, LTeamID)
        values = np.append(values, WTeamID)
        values = np.append(values, loc)
        
    # add winnning team position to data
    difference = np.append(difference, winner)
    difference = np.round(difference, decimals = 3)
    
    # merge descriptive data with team data
    instance = np.concatenate([values, difference])
    
    # add game statistics to dataset
    dataset[i] = instance


# -------------------------
# DATA STORAGE
# ------------------------- 
np.savetxt("/kaggle/working/all_dataset.csv", dataset, delimiter = ",")

# output feedback
print()
print()
print("Training Dataset creation complete.")
print()
print()
# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data 
data = pd.read_csv('/kaggle/working/all_datasetR.csv', header = None)
X = data.iloc[:, 4:-1].values
y = data.iloc[:, -1].values

# create a train-test split with ratio 15:85
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# scale the training and testing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# save the scaler for future use
scalerfile = '/kaggle/working/scaler.save'
pickle.dump(sc, open(scalerfile, 'wb'))


# -------------------------
# MODEL CREATION
# -------------------------
# 1 - logistic regression
print("Training Logistic Regression model...")

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(solver = 'lbfgs')
classifier_lr.fit(X_train, y_train)

y_pred = classifier_lr.predict(X_test)
from sklearn.metrics import accuracy_score
score_lr = accuracy_score(y_test, y_pred)

# 2 - random forest
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 75, criterion = 'entropy')
classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)
from sklearn.metrics import accuracy_score
score_rf = accuracy_score(y_test, y_pred)

# 3 - naive bayes
print("Training Naive Bayes model...")
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred = classifier_nb.predict(X_test)
from sklearn.metrics import accuracy_score
score_nb = accuracy_score(y_test, y_pred)

# 4 - neural network
print("Training Neural Network model...")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

classifier_nn = Sequential()
classifier_nn.add(Dense(50, input_dim = X_train.shape[1], 
                kernel_initializer = 'random_uniform', 
                activation = 'sigmoid'))
classifier_nn.add(Dropout(0.2))
classifier_nn.add(Dense(100, activation = 'relu'))
classifier_nn.add(Dropout(0.5))
classifier_nn.add(Dense(100, activation = 'relu'))
classifier_nn.add(Dropout(0.5))
classifier_nn.add(Dense(25, activation = 'relu'))
classifier_nn.add(Dropout(0.2))
classifier_nn.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))

adam = optimizers.Adam(lr = 0.005)
classifier_nn.compile(loss = 'binary_crossentropy', optimizer = adam)
classifier_nn.fit(X_train, y_train, batch_size = 20, epochs = 5)

y_pred = classifier_nn.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
from sklearn.metrics import accuracy_score
score_nn = accuracy_score(y_test, y_pred)


# -------------------------
# MODEL EVALUATION
# -------------------------
# print the accuracy score of each 
print()
print()
print("Logisitc Regression Accuracy: " + str(round(score_lr, 3)))
print("Random Forest Accuracy: " + str(round(score_rf, 3)))
print("Naive Bayes Accuracy: " + str(round(score_nb, 3)))
print("Neural Network Accuracy: " + str(round(score_nn, 3)))


# -------------------------
# MODEL STORAGE
# -------------------------
# identify the model with the highest accuracy score
classifiers = [classifier_lr, classifier_rf, classifier_nb, classifier_nn]
scores = [score_lr, score_rf, score_nb, score_nn]
x = scores.index(max(scores))

# store the model with the highest accuracy score to disk
with open('/kaggle/working/predictor.pkl', 'wb') as fid:
    pickle.dump(classifiers[x], fid)

# output feedback
print()
print()
print("Predictive Models creation complete.")
print()
print()