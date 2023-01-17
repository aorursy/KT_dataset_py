import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline
df=pd.read_csv('../input/voice.csv')
df.head()
df.info()
df.describe()
df['label'].value_counts()
le=LabelEncoder()
le.fit_transform(df['label'])
correlation = df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, square=True, annot=True ,cmap='coolwarm')
features=df.drop(['label','centroid','dfrange'],axis=1)
target=df['label']

sc=StandardScaler()
sc.fit_transform(features)
(X_train,X_test,Y_train,Y_test)=train_test_split(features,target,test_size=0.30)
mlp_classifier= MLPClassifier(activation='tanh',hidden_layer_sizes=(8,4),max_iter=2000,solver='lbfgs',random_state=1)
mlp_classifier.fit(X_train,Y_train)
mlp_classifier.score(X_test,Y_test)
