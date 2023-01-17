import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
train_df = pd.read_csv('../input/train/train.csv')
train_df.head(3)
train_df.describe()
import missingno as msno
msno.bar(train_df,figsize=(20,4))
plt.style.use('ggplot')
train_df.AdoptionSpeed.value_counts().plot(kind='bar')
train_df.Type.value_counts().plot(kind='bar')
breed_label = pd.read_csv('../input/breed_labels.csv')
breed_label.head()
train_df.Breed1.value_counts().reset_index().join(breed_label.set_index('BreedID'),on='index').rename(columns={'index':'Breed1','Breed1':'Count'}).tail()
train_df.Breed2.value_counts().reset_index().join(breed_label.set_index('BreedID'),on='index').rename(columns={'index':'Breed2','Breed2':'Count'}).tail()
train_df.Breed2.loc[train_df.Breed1==train_df.Breed2] = 0
train_df.Breed2.loc[train_df.Breed1==train_df.Breed2]
train_df['Mixed_Breed'] = train_df.apply(lambda x: 0 if x.Breed2==0 and x.Breed1!=307 else 1, axis=1)
train_df[train_df["Breed2"]!=0].head(3)
train_df.Mixed_Breed.value_counts().plot(kind='bar')
color_label = pd.read_csv('../input/color_labels.csv')
color_label
train_df.Color1.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color1','Color1':'Count'})
train_df.Color2.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color2','Color2':'Count'})
train_df.Color3.value_counts().reset_index().join(color_label.set_index('ColorID'),on='index').rename(columns={'index':'Color3','Color3':'Count'})
train_df['Num_Color'] = train_df.apply(lambda x:  3-sum([y==0 for y in [x.Color1, x.Color2, x.Color3]]), axis=1)
train_df.Num_Color.value_counts().plot(kind='bar')
train_df.MaturitySize.value_counts().plot(kind='bar')
state_label = pd.read_csv('../input/state_labels.csv')
state_label
train_df.State.value_counts().reset_index().join(state_label.set_index('StateID'),on='index').rename(columns={'index':'State','State':'Count'})
train_df['Description'].fillna("", inplace=True)
train_df['Description_Length'] = train_df.Description.map(len)
plt.figure(figsize=(20,10))
sns.boxplot(x='AdoptionSpeed', y='Description_Length', data=train_df, showfliers=False)
y = train_df['AdoptionSpeed']
train_df.info()
x = train_df.drop(["Name","RescuerID","Description","PetID", "AdoptionSpeed"], axis = 1)
x.head(3)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)

#sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize=(6,6))
y_pos = np.arange(x.shape[1])
plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
plt.yticks(y_pos, x)
plt.xlabel('features')
plt.title('feature_importances')
plt.show()
from sklearn.decomposition import PCA
pca = PCA(10)
newdata = pca.fit_transform(x)
newdata.shape
pca.explained_variance_ratio_      # 百分比
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_pca = pca.fit_transform(ss.fit_transform(x))
x_pca.shape
y.shape
y2 = y.values
from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=100,
                             min_samples_split=12, #20
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(x, y2) #filter SP data
print("Out Of Bag score is %.4f" % rf.oob_score_)
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets with 20% test rate
X_train, X_test, y_train, y_test = train_test_split(x, y2, test_size = 0.2, random_state = 0)
# Training model
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=101)
RFC.fit(X_train,y_train)

# Import 4 metrics from sklearn for testing
from sklearn.metrics import accuracy_score
print ("Accuracy on testing data of RandomForestClassifier: {:.4f}".format(accuracy_score(y_test, RFC.predict(X_test))))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, RFC.predict(X_test))
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10, 50, 100, 200, 400],
              'min_samples_split':[8,12,16,20],
              'min_samples_leaf':[1,2,3,4,5]
             }
#grid = GridSearchCV(rf, parameters)
#grid_fit = grid.fit(x, y2)
# Get the estimator
#best_rf = grid_fit.best_estimator_
# Make predictions using the unoptimized and model
#predictions_rf = (rf.fit(X_train, y_train)).predict(X_test)
#best_predictions_rf = best_rf.predict(X_test)

#from sklearn.metrics import accuracy_score
#print ("Accuracy on testing data of RandomForestClassifier: {:.4f}".format(accuracy_score(y_test, best_predictions_rf)))
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, best_predictions_rf)