import pandas as pd
import numpy as np
fifa = pd.read_csv("../input/complete.csv")
fifa = fifa.rename(columns={'age': 'position'})
cb = pd.DataFrame(fifa.loc[(fifa["prefers_cb"]==True)])
cb['position'] = 0

wb = pd.DataFrame(fifa.loc[(fifa["prefers_rb"]==True)|(fifa["prefers_lb"]==True)|(fifa["prefers_rwb"]==True)|(fifa["prefers_lwb"]==True)])
wb['position'] = 1

mid = pd.DataFrame(fifa.loc[(fifa["prefers_cdm"]==True)|(fifa["prefers_cm"]==True)|(fifa["prefers_cam"]==True)])
mid['position'] = 2

wingers = pd.DataFrame(fifa.loc[(fifa["prefers_lm"]==True)|(fifa["prefers_rm"]==True)|(fifa["prefers_lw"]==True)|(fifa["prefers_rw"]==True)])
wingers['position'] = 3

st = pd.DataFrame(fifa.loc[(fifa["prefers_st"]==True)|(fifa["prefers_cf"]==True)])
st['position'] = 4
data = [cb,wb,st,mid, wingers]
result = pd.concat(data)
result = result.sort_values('overall', ascending=False)
result = result.drop_duplicates('name')
fifa = result.loc[:,["name","club_logo","flag","photo" ,"acceleration",
                       "sprint_speed", "positioning","finishing","shot_power",
                       "long_shots","volleys","penalties","vision","crossing",
                       "free_kick_accuracy","short_passing","long_passing","curve",
                       "agility","balance","reactions","ball_control","dribbling",
                       "composure","interceptions","heading_accuracy","marking",
                       "standing_tackle","sliding_tackle","jumping","stamina",
                       "strength","aggression", "position"]]
data = fifa.iloc[:, 4:33]
labels = fifa["position"]
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

features_train, features_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.3, random_state=42)

rnd_clf = RandomForestClassifier(n_estimators = 500,criterion = 'entropy', min_samples_split = 2,min_samples_leaf = 5, max_depth=20)
rnd_clf.fit(features_train, labels_train)
pred = rnd_clf.predict(features_test)
print(accuracy_score(labels_test,pred))
target_names = ['cb', 'wb', 'mid','wing','st']
print(classification_report(labels_test,pred, target_names = target_names))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(.5, 1))

features = scaler.fit_transform(rnd_clf.feature_importances_.reshape(-1,1))
i = 0
color_list = []
while i<len(features):
    #pace colors
    if i<2:
        arr = (features[i],features[i]*0,features[i]*.4)
        color_list.append(arr)
    #shooting colors
    elif i<8:
        arr = (features[i],features[i],features[i]*0)
        color_list.append(arr)
    #passing colors
    elif i<14:
        arr = (features[i]*.1,features[i],features[i])
        color_list.append(arr)
    #dribbling colors
    elif i<20:
        arr = (features[i]*.5,features[i]*0,features[i])
        color_list.append(arr)
    #defending colors
    elif i<25:
        arr = (features[i],features[i]*.3,features[i]*.0)
        color_list.append(arr)
    #physical colors
    elif i<29:
        arr = (features[i]*0,features[i]*.8,features[i]*0)
        color_list.append(arr)
    i+=1
color_list = np.array(color_list)
color_list =color_list.reshape(len(rnd_clf.feature_importances_),3) 
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt 

width = 0.5
graph_labels = data.columns.values

fig, ax = plt.subplots(figsize=(20, 10))

ind = np.arange(len(graph_labels))
test = ax.bar(ind, rnd_clf.feature_importances_, width, align='center', alpha=0.5, color = color_list)
ax.set_xticks(ind)
ax.set_xticklabels(graph_labels,rotation=90, fontsize=15)
ax.set_ylabel('Importance',fontsize=20)
ax.set_title('Importance of ingame stats in position classification',fontsize=20)
plt.show()
from sklearn.neighbors import KNeighborsClassifier

knn_clf=KNeighborsClassifier(n_neighbors=30, weights ='distance')
knn_clf.fit(features_train, labels_train)
pred = knn_clf.predict(features_test)
print(accuracy_score(labels_test,pred))
print(classification_report(labels_test,pred))
knn_clf_2=KNeighborsClassifier(n_neighbors=30, weights ='distance')
knn_clf_2.fit(data, labels)
from IPython.display import HTML, display

player_index = 212
table = "<table><tr><td><b>{}</b></td><td><img src={}></td><td><img src={}></td><td><img src={}></td></tr></table>"

neighbors = knn_clf_2.kneighbors([data.iloc[player_index,:]],n_neighbors=5)[1]
neighbors = np.reshape(neighbors,(neighbors.shape[1]))
display(HTML(table.format(fifa.iloc[player_index,[0]].values,fifa.iloc[player_index,3],
                          fifa.iloc[player_index,1],fifa.iloc[player_index,2])))
print("--------------Neighbors--------------")

i = 1
while i<len(neighbors):
    display(HTML(table.format(fifa.iloc[neighbors[i],[0]].values,fifa.iloc[neighbors[i],3],
                              fifa.iloc[neighbors[i],1],fifa.iloc[neighbors[i],2])))
    i+=1
