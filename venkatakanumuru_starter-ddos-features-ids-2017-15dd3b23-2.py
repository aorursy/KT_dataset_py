import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report



df_l = pd.read_csv('../input/alldays_ddos.csv', dtype='unicode')



df_b = df_l.fillna(0)





df_b1 = pd.DataFrame(df_b)

mymap = {'BENIGN': 1, 'SSH-Patator': 2, 'DoS slowloris' : 3, 'DoS Slowhttptest' : 4, 'FTP-Patator' : 5, 'DoS Hulk' : 6, 'DoS GoldenEye': 7, 'Heartbleed': 8, 'Web Attack – Brute Force': 9, 'Web Attack – XSS': 10, 'Web Attack – Sql Injection': 11, 'Infiltration': 12, 'Bot': 13, 'DDoS': 14, 'PortScan': 15}

df = df_b1.applymap(lambda s: mymap.get(s) if s in mymap else s)





x = df.iloc[:, :-1].values

y = df.iloc[:, 5].values





X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)





from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



X_train = pd.DataFrame(X_train)

y_train = pd.DataFrame(y_train)

X_test = pd.DataFrame(X_test)

y_test = pd.DataFrame(y_test)



X_train = X_train.fillna(0)

y_train = y_train.fillna(0)

X_test = X_test.fillna(0)

y_test = y_test.fillna(0)



X_train=X_train.astype('int')

y_train=y_train.astype('int')

X_test=X_test.astype('int')

y_test=y_test.astype('int')





classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train.values.ravel())





y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred))
