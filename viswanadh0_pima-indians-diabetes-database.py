import pandas as pd
import numpy as np

main_df = pd.read_csv('../input/diabetes.csv')
main_df.head()
main_df.shape
main_df.describe()
main_df["Outcome"].value_counts()
main_df.columns
import matplotlib.pyplot as plt
import seaborn as sns
sns.FacetGrid(main_df, hue='Outcome', size= 6) \
    .map(plt.scatter, 'Glucose', 'Age', ) \
    .add_legend()
plt.show()
sns.boxplot(x='Outcome', y='Glucose', data=main_df)
plt.show()
from sklearn import model_selection

x = main_df.ix[:,0:8]
y = main_df['Outcome']

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x,y, test_size=0.2)

x.shape
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_pred, Y_test)
from sklearn.ensemble import RandomForestClassifier

clf_randomForest = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0)
clf_randomForest.fit(X_train, Y_train)
y_rf_pred = clf_randomForest.predict(X_test)
accuracy_score(y_rf_pred, Y_test)
