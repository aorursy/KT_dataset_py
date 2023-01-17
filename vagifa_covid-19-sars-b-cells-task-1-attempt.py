import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



train = pd.read_csv('/kaggle/input/epitope-prediction/input_bcell.csv')

test = pd.read_csv('/kaggle/input/epitope-prediction/input_sars.csv')
sns.set_style('darkgrid')
merged = pd.concat([train,test],axis=0,ignore_index=True)
merged.head(5)
merged.info()
merged.describe()
merged.isnull().sum()
sns.countplot(merged['target'])

plt.show()
plt.figure(figsize=(12,12))

sns.heatmap(merged.corr(),annot=True,cmap='coolwarm')

plt.show()
merged.hist(figsize=(15,15))

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupKFold

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
# Feature Preprocessing function



# Scale function    

def scale(data):

    scaler = StandardScaler()

    data_scaled = scaler.fit_transform(data)

    return data_scaled



    

# Feature Engineering functions       

def get_length(df):

    df['length'] = df['end_position'] - df['start_position']+1

    

# Using KMeans clustering and PCA to generate new features

def kmeans_features(X,X_test):

    X_scaled,X_test_scaled = scale(X),scale(X_test)

    pca = PCA(n_components=2)

    X_pca_train = pca.fit_transform(X_scaled)

    X_pca_test = pca.fit_transform(X_test_scaled)

    

    kmeans_train = KMeans(n_clusters=4,max_iter=500,random_state=42)

    kmeans_train.fit(X_pca_train)



    kmeans_test = KMeans(n_clusters=4,max_iter=500,random_state=42)

    kmeans_test.fit(X_pca_test)

    

    X = pd.DataFrame(X)

    X['kmeans_feature'] = kmeans_train.labels_

    

    X_test = pd.DataFrame(X_test)

    X_test['kmeans_feature'] = kmeans_test.labels_

    

    return X,X_test
get_length(train)

get_length(test)
features = ["chou_fasman","emini","kolaskar_tongaonkar","parker","length","isoelectric_point","aromaticity","hydrophobicity","stability"]



X,y = train[features],train['target']

X_test,y_test = test[features],test['target']
gkf = GroupKFold(n_splits=5)
X,X_test = kmeans_features(X,X_test)
for train_index,test_index in gkf.split(X,y,train['parent_protein_id']):

    X_train,X_valid = X.iloc[train_index],X.iloc[test_index],

    y_train,y_valid = y[train_index],y[test_index]
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,roc_curve

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=42,max_iter=400,hidden_layer_sizes=(100,))
mlp.fit(X_train,y_train)
def visual_evaluation(model,gradient_booster=True):    

    plt.figure(figsize=(10,10))

    sns.heatmap(confusion_matrix(y_test,model.predict(X_test)),cmap='coolwarm',annot=True)

    plt.show()

    

    fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))

    plt.plot(fpr,tpr,linestyle='--')

    plt.show()
def text_evaluation(model):

    print(classification_report(y_test,model.predict(X_test)))

    print("Accuracy: " + str(accuracy_score(y_test,model.predict(X_test))))

    print('AUC Score: ' + str(roc_auc_score(y_test,model.predict(X_test))))
visual_evaluation(mlp,gradient_booster=False)
text_evaluation(mlp)