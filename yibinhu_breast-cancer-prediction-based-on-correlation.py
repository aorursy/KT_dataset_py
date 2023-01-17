import pandas as pd, numpy as np
df = pd.read_csv('../input/data.csv')

df.info()
cleaned_data = df.iloc[:,1:-1]

cleaned_data.info()
cleaned_data.head()

# diagnosis is a categorial column, so encode this

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()

cleaned_data.iloc[:, 0] = label_encoder.fit_transform(cleaned_data.iloc[:,0]).astype('float64')

cleaned_data.head()
corr = cleaned_data.corr()

import seaborn as sns

sns.heatmap(corr)
columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = cleaned_data.columns[columns]

data = cleaned_data[selected_columns]

print('selected columns are : {}'.format(selected_columns))

#print('data head : {}'.format(data.head()))
# store result

result = pd.DataFrame()

result['diagnosis'] = data.iloc[:,0]



# visualizing selected features, this function is quoted from towarddatascience.

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (20,25))

j = 0

for i in data.columns:

    plt.subplot(6, 4, j+1)

    j += 1

    sns.distplot(data[i][result['diagnosis']==0], color='g', label = 'benign')

    sns.distplot(data[i][result['diagnosis']==1], color='r', label = 'malignant')

    plt.legend(loc='best')

fig.suptitle('Breast Cance Data Analysis')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
# split data into training set and test set

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(data.values, result, test_size = 0.2)
svc = SVC()

lsvc = LinearSVC()

lsvc.fit(X_train, y_train)

pred = lsvc.predict(X_test)

cm = confusion_matrix(y_test, pred)

print(lsvc.score(X_train, y_train))

#sum = 0

#for i in range(cm.shape[0]):

#    sum += cm[i][i]

    

#accuracy = sum/X_test.shape[0]

#print(accuracy)
