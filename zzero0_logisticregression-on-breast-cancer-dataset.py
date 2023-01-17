import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df =pd.read_table('../input/breast-cancer-wisconsin.data.txt', delimiter=',', names=('id number','clump_thickness','cell_size_uniformity','cell_chape_uniformity','marginal_adhesion','epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'))
df.head()
df.info()
df["bare_nuclei"] = df["bare_nuclei"][df["bare_nuclei"]!='?']

#removed_those_rows_for_which_bare_nuclei_=_'?'
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis') #visualizing missing_data
df.dropna(inplace=True)
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis') #no_missing_data
df.info()
df["bare_nuclei"] = df["bare_nuclei"].astype("int64")
from sklearn.model_selection import train_test_split
pd.get_dummies(df["class"]).head()
df["class"] = pd.get_dummies(df["class"],drop_first=True) #"class" column had 2 values - 2 for benign, 4 for malignant
X_train, X_test, y_train, y_test = train_test_split(df[df.columns[1:-1]], df["class"], test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=100)
logreg.fit(X_train,y_train)
pred = logreg.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
logreg.score(X_train,y_train)
logreg.score(X_test,y_test)
cm
df_cm = pd.DataFrame(cm,index = ['AN','AP'],columns=['PN','PP'])
sns.heatmap(df_cm,cbar=True,cmap='viridis') #AN-actually negative AP-actually positive PN- predicted neagative PP- predicted positive