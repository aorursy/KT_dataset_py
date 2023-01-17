# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/games.csv")
df
df.columns
df = df[["winner","firstBlood","firstTower","firstInhibitor","firstBaron","firstDragon","firstRiftHerald","t1_towerKills","t1_inhibitorKills",'t1_baronKills',
       't1_dragonKills','t2_towerKills','t2_inhibitorKills', 't2_baronKills', 't2_dragonKills'
       ]]
df
org_df = pd.read_csv("../input/games.csv")
plt.figure(figsize=(12,6))
p1=sns.kdeplot(org_df['gameDuration'], shade=True, color="r").set_title('Distribution of Duration')
num_of_missing = []
for i in df.columns:
    missing = sum(df[i].isna())
    num_of_missing.append(missing)
num_of_missing
import matplotlib.pyplot as plt
plt.hist(df["t1_towerKills"])
plt.show()
plt.hist(df["t2_towerKills"])
plt.show()
(df["t1_towerKills"].mean(),df["t2_towerKills"].mean())
# when team 1 get the first blood and win the game
f1w1 = len(df.loc[(df['winner']==1) & (df['firstBlood']==1)])/len(df)
# when team 1 get the first blood and team 2 wins
f1w2 = len(df.loc[(df['winner']==2) & (df['firstBlood']==1)])/len(df)

height1 = [f1w1,f1w2]
bars1 = ("team1","team2")
y_pos1 = np.arange(len(bars1))
plt.bar(y_pos1,height1)
plt.title("winning rate when team 1 got first blood")
plt.xticks(y_pos1, bars1)
plt.show()
print(abs(f1w1-f1w2))

# when team 2 get the first blood and win the game
f2w2 = len(df.loc[(df['winner']==2) & (df['firstBlood']==2)])/len(df)
# when team 2 get the first blood and team 1 wins
f2w1 = len(df.loc[(df['winner']==1) & (df['firstBlood']==2)])/len(df)

height2 = [f2w1,f2w2]
bars2 = ("team1","team2")
y_pos2 = np.arange(len(bars2))
plt.bar(y_pos2,height2)
plt.title("winning rate when team 2 got first blood")
plt.xticks(y_pos2, bars2)
plt.show()
print(abs(f2w1-f2w2))
# when team 1 get the first tower and win the game
t1w1 = len(df.loc[(df['winner']==1) & (df['firstTower']==1)])/len(df)
# when team 1 get the first tower and team 2 wins
t1w2 = len(df.loc[(df['winner']==2) & (df['firstTower']==1)])/len(df)

height3 = [t1w1,t1w2]
bars3 = ("team1","team2")
y_pos3 = np.arange(len(bars3))
plt.bar(y_pos3,height3)
plt.title("winning rate when team 1 got first tower")
plt.xticks(y_pos3, bars3)
plt.show()
print(abs(t1w1-t1w2))

# when team 2 get the first tower and win the game
t2w2 = len(df.loc[(df['winner']==2) & (df['firstTower']==2)])/len(df)
# when team 2 get the first tower and team 1 wins
t2w1 = len(df.loc[(df['winner']==1) & (df['firstTower']==2)])/len(df)

height4 = [t2w1,t2w2]
bars4 = ("team1","team2")
y_pos4 = np.arange(len(bars4))
plt.bar(y_pos4,height4)
plt.title("winning rate when team 2 got first tower")
plt.xticks(y_pos4, bars4)
plt.show()
print(abs(t2w1-t2w2))
# when team 1 get the first baron and win the game
t4w1 = len(df.loc[(df['winner']==1) & (df['firstBaron']==1)])/len(df)
# when team 1 get the first baron and team 2 wins
t4w2 = len(df.loc[(df['winner']==2) & (df['firstBaron']==1)])/len(df)

height3 = [t4w1,t4w2]
bars3 = ("team1","team2")
y_pos3 = np.arange(len(bars3))
plt.bar(y_pos3,height3)
plt.title("winning rate when team 1 got first baron")
plt.xticks(y_pos3, bars3)
plt.show()

# when team 2 get the first baron and win the game
t3w2 = len(df.loc[(df['winner']==2) & (df['firstBaron']==2)])/len(df)
# when team 2 get the first baron and team 1 wins
t3w1 = len(df.loc[(df['winner']==1) & (df['firstBaron']==2)])/len(df)

height4 = [t3w1,t3w2]
bars4 = ("team1","team2")
y_pos4 = np.arange(len(bars4))
plt.bar(y_pos4,height4)
plt.title("winning rate when team 2 got first baron")
plt.xticks(y_pos4, bars4)
plt.show()
game = df.copy()
y = game["winner"].values
x = game.drop(["winner"],axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(x_train,y_train)
pred = clf1.predict(x_test)
pred
clf1.score(x_test,y_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
criterion=["gini","entropy"]
max_depth=range(1,20,2)
splitter=["best","random"]
dt=DecisionTreeClassifier()
grid_decision_tree=GridSearchCV(estimator=dt,cv=15,param_grid=dict(criterion=criterion,max_depth=max_depth,splitter=splitter))
grid_decision_tree.fit(x_train,y_train)
print(grid_decision_tree.best_score_)
print(grid_decision_tree.best_params_)
criterion=["gini","entropy"]
max_depth=range(1,20,2)
from sklearn.ensemble import RandomForestClassifier
gini_score = []
entropy_score = []
for i in max_depth:
    clf2 = RandomForestClassifier(max_depth=i,criterion="gini")
    clf2.fit(x_train,y_train)
    sc = clf2.score(x_test,y_test)
    gini_score.append(sc)
for i in max_depth:
    clf2 = RandomForestClassifier(max_depth=i,criterion="entropy")
    clf2.fit(x_train,y_train)
    sc = clf2.score(x_test,y_test)
    entropy_score.append(sc)
print(gini_score)
print(entropy_score)
(max(gini_score),max(entropy_score))
max_depth[4]
from sklearn.neighbors import KNeighborsClassifier
n_neighbors = [5,7,9,11]
weights=["uniform","distance"]
algorithm = ["auto","brute"]
knn = KNeighborsClassifier()
grid_KNN = GridSearchCV(estimator=knn,cv=15,param_grid=dict(n_neighbors=n_neighbors,weights=weights,algorithm=algorithm))
grid_KNN.fit(x_train,y_train)
print(grid_KNN.best_score_)
print(grid_KNN.best_params_)
from sklearn.svm import SVC
kernel = ["linear", "poly", "rbf", "sigmoid"]
svc = SVC()
grid_svc = GridSearchCV(estimator=svc,cv=15,param_grid=dict(kernel=kernel))
grid_svc.fit(x_train,y_train)
print(grid_svc.best_score_)
print(grid_svc.best_params_)
final_clf = RandomForestClassifier(max_depth=9,criterion="gini")
final_clf.fit(x_train,y_train)
print("score:", final_clf.score(x_test,y_test))
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(final_clf.estimators_[0], out_file='tree.dot', 
                feature_names = x.columns,
                class_names = ["1","2"],
                rounded = True,
                filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
Image(filename = 'tree.png')
from sklearn.metrics import confusion_matrix,classification_report
predicted_values = final_clf.predict(x_test)
cm=confusion_matrix(y_test,predicted_values)
cr=classification_report(y_test,predicted_values)
print('Classification report : \n',cr)
import seaborn as sns
g1 = sns.heatmap(cm,annot=True,fmt=".1f",cmap="flag",cbar=False)
g1.set_ylabel('y_true')
g1.set_xlabel('y_head')
scenario={"feature":["first_blood","first_tower","first_inhibitor","first_Baron","first_Dragon","first_RiftHerald",
"t1_tower","t1_inhibitor","t1_baron","t1_dragon","t2_tower","t2_inhibitor","t2_baron","t2_dragon"],
         "value":[2,0,1,2,2,1,10,2,1,3,5,2,0,1]}
scen=pd.DataFrame(scenario)
scen.T
x1=[[1,1,2,1,1,1,10,2,1,4,7,2,1,1]]
c=final_clf.predict_proba(x1).reshape(-1,1)
print("winner is :" , final_clf.predict(x1) )
print("first team win probability is % ", list(c[0]*100),"\nsecond team win probability is %:",list(c[1]*100)  )
