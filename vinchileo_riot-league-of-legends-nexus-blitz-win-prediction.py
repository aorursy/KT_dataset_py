# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#START
import json
#load players and cashedmatches
file = open('../input/cashedmatches-set.txt', 'r')
cashedmatches=json.load(file)
file.close()
print('all cashed matches',len(cashedmatches))
file = open('../input/players-set.txt', 'r')
players=set(json.load(file))
file.close()
print('all players in cashedmatches:',len(players))
#convert to numpy array, find champions array
m = pd.DataFrame(cashedmatches)
x = m.iloc[:,13:23].copy()
champions = np.unique(x[:])
print(champions)
print(type(champions))
m.head(5)
#Calculate players win stats for all type of champions

cdict = {i:[0,0,0] for i in list(champions)} #win, fail, winrate
#print(cdict)
#2 - win 100 blue or 200 red 
#3-12 players
#13-22 champions
#25-35 rank
print(type(players))
players_stat={id:{i:[0,0,0] for i in list(champions)} for id in list(players)}
#print(len(players_stat),players_stat[202054882])
#for match in riot.cashedmatches:
for match in cashedmatches:
    #print(match[3:23])
    win = match[2]
    for i in range(3,13):
        playerid = match[i]
        champion = match[i+10]
        if win==100:#blue win
            if i<=7:
                index=0 #win
            else:
                index=1 #fail
        else:
            if i<=7:
                index=1 #fail
            else:
                index=0 #win
        last = players_stat[playerid][champion]
        #print()
        #print(i, playerid, win, index, champion, last)
        players_stat[playerid][champion][index] = last[index] + 1
        #print(i, playerid, win, index, champion, players_stat[playerid][champion])
        #print(players_stat[player])
#print(players_stat[player])

#players_stat:
print(len(players_stat),players_stat[202054882])
#calculate players common winrate
players_winrate={}
for player,c in players_stat.items():
    winrate=[0,0,0]
    #print(player,champions)
    for index in c.values():
        if (index[0]+index[1]) > 0:
            index[2] = index[0]/(index[0]+index[1])
        winrate[0]+=index[0]
        winrate[1]+=index[1]
    winrate[2]=winrate[0]/(winrate[0]+winrate[1])
    players_winrate[player]=winrate
print(len(players_winrate))
#only winrate (30 features = 10 players *3)
#MSE: 0.0703
#Accuracy score (training): 0.953
#Accuracy score (validation): 0.930
newmatches = []
#for match in riot.cashedmatches:
for match in cashedmatches:
    a=[match[2]/100-1]
    for i in range(3,13): #get players
        a.extend(players_winrate[match[i]])
    newmatches.append(a)
print(len(newmatches))
print(newmatches[0])
#matches preprocessing with Standart scaling and split dataset 70% - train, 30% - test
matches = pd.DataFrame(newmatches)
x = matches.iloc[:,1:]
y = matches[[0]]
y1d = y.values.ravel()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
print(scaler.mean_)
print(scaler.scale_)
#x_scaled = scaler.transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y1d, test_size=0.3) #x_scaled
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

#transform:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#Use GBC for predictions
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
#from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# #############################################################################
# Fit regression model
#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01}
params =  {'n_estimators': 300, 'learning_rate':0.5, 'max_features':2, 'max_depth': 2}#, 'random_state': 0}
clf = ensemble.GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("MSE: %.4f" % mse)
print("Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))
print()
    
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print()
print("Classification Report")
print(classification_report(y_test, predictions))

y_scores_clf = clf.decision_function(X_test)
fpr_clf, tpr_clf, _ = roc_curve(y_test, y_scores_clf)
roc_auc_clf = auc(fpr_clf, tpr_clf)

print("Area under ROC curve = {:0.2f}".format(roc_auc_clf))
# #############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.show()
# #############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
print(sorted_idx)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)
plt.barh(sorted_idx, feature_importance[sorted_idx], align='center')
plt.yticks(pos, x.iloc[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, x.iloc[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
import tensorflow as tf
from tensorflow import keras
#print(tf.__version__)
#print(tf.__git_version__,tf.__compiler_version__,tf.__cxx11_abi_flag__,tf.__monolithic_build__)
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)
'''
l2_bigger_model = keras.models.Sequential([
    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01),
activation=tf.nn.relu, input_shape=(30,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, kernel_regularizer=keras.regularizers.l2(0.01),
activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
l2_bigger_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy','binary_crossentropy'])
l2_bigger_model.summary()
l2_bigger_history = l2_bigger_model.fit(X_train, y_train,
                                epochs=20,
                                batch_size=1024,
                                validation_data=(X_test, y_test),
                                verbose=1)
def plot_history(histories, key='binary_crossentropy'): # , acc
  plt.figure(figsize=(20,5))
    
  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

plot_history([('l2_bigger', l2_bigger_history)], key='binary_crossentropy')
plot_history([('l2_bigger', l2_bigger_history)], key='acc')
plt.show()
predict  = l2_bigger_model.predict(X_test,verbose=1)
print(predict.mean(), predict.std(), predict.var())
train_predict = l2_bigger_model.predict(X_train,verbose=1)
print(train_predict.mean(), train_predict.std(), train_predict.var())
#BETTING TEST
balance = 100
bet = 0.5 #% from balance
win = 1.2 #win reward multiplied
predict  = l2_bigger_model.predict(X_test,verbose=1)
print(len(predict), len(y_test))
count = 0
for i in range(len(predict)):
    if abs(y_test[i]-predict[i])<0.5:
        count+=1
print(count, count/len(y_test))

for i in range(100):
#for i in range(len(predict)):
    if abs(0.5-predict[i])>0.49: #prediction >0.99%
        cash = balance * bet
        balance-=cash
    else:
        continue
    if abs(y_test[i]-predict[i])<0.5:
        reward = win * cash
        balance +=reward
    print(balance)
print(balance)