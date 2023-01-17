import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/knn-project-data/KNN_Project_Data')
df.head()
sns.pairplot(data=df, palette='viridis')
from sklearn.preprocessing import StandardScaler
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_feat, df['TARGET CLASS'], test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
error_data = []

for i in range(1,50):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    predictions = knn_model.predict(X_test)
    error_data.append(np.mean(y_test != predictions))
error_data
plt.figure(figsize=(10,4))
plt.plot(range(1,50), error_data, linestyle='--', markerfacecolor='red', markersize=10, marker='o')
knn_model = KNeighborsClassifier(n_neighbors=42)
knn_model.fit(X_train, y_train)
predictions = knn_model.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))