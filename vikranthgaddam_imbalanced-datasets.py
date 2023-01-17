import numpy as np

import pandas as pd



train = pd.read_csv('../input/testing/c_4.csv')



target_count = train.A64.value_counts()

print('Class 3:', target_count[3])

print('Class 4:', target_count[4])

print('Proportion:', round(target_count[4] / target_count[3], 2), ': 1')



target_count.plot(kind='bar', title='Count (target)');
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

labels = train.drop(['A64'],axis=1)

y = train['A64']



X_train, X_test, y_train, y_test = train_test_split(labels, y, test_size = 0.3, random_state = 0) 



print("Number transactions X_train dataset: ", X_train.shape) 

print("Number transactions y_train dataset: ", y_train.shape) 

print("Number transactions X_test dataset: ", X_test.shape) 

print("Number transactions y_test dataset: ", y_test.shape) 
from sklearn.svm import SVC # "Support Vector Classifier" 

classifier_linear = SVC(kernel='linear',gamma='auto', random_state = 1)

classifier_linear.fit(X_train,y_train)

Y_linear = classifier_linear.predict(X_test)

results_=confusion_matrix(Y_linear,y_test)

print(results_)
from sklearn.metrics import classification_report

target_names=['class 3','class 4']



print(classification_report(y_test,Y_linear, target_names=target_names))
print("Before OverSampling, counts of label '3': {}".format(sum(y_train == 3))) 

print("Before OverSampling, counts of label '4': {} \n".format(sum(y_train == 4))) 

  

from imblearn.over_sampling import SMOTE 

sm = SMOTE(kind='regular',k_neighbors=2) 

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 



  

print("After OverSampling, counts of label '3': {}".format(sum(y_train_res == 3))) 

print("After OverSampling, counts of label '4': {}".format(sum(y_train_res == 4))) 

from sklearn.svm import SVC # "Support Vector Classifier"

labels1 = train.drop(['A64'],axis=1)

classifier_linear = SVC(kernel='linear',gamma='auto', random_state = 1)

classifier_linear.fit(X_train_res,y_train_res)

Y_linear = classifier_linear.predict(X_test)

results_=confusion_matrix(Y_linear,y_test)

print(results_)
from sklearn.metrics import classification_report

target_names=['class 3','class 4']



print(classification_report(y_test,Y_linear, target_names=target_names))
print(classification_report(y_test, Y_linear))
# Class count

count_class_4, count_class_3 = train.A64.value_counts()



# Divide by class

df_class_3 = train[train['A64'] == 3]

df_class_4 = train[train['A64'] == 4]
df_class_4_under = df_class_4.sample(count_class_3)

df_test_under =pd.concat([df_class_4_under, df_class_3], axis=0)



print('Random under-sampling:')

print(df_test_under.A64.value_counts())



df_test_under.A64.value_counts().plot(kind='bar', title='Count (target)');

classifier_linear = SVC(kernel='linear',gamma='auto', random_state = 1)



classifier_linear.fit(X_train, y_train)



y_pred = classifier_linear.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
df_class_3_over = df_class_3.sample(count_class_4, replace=True)

df_test_over = pd.concat([df_class_4, df_class_3_over], axis=0)



print('Random over-sampling:')

print(df_test_over.A64.value_counts())



df_test_over.A64.value_counts().plot(kind='bar', title='Count (target)');
correlations_data =train.corr()['A64']

print(correlations_data)



for x in correlations_data:

    print(x)
import imblearn
from sklearn.datasets import make_classification



X, y = make_classification(

    n_classes=2, class_sep=1.5, weights=[0.9, 0.1],

    n_informative=3, n_redundant=1, flip_y=0,

    n_features=20, n_clusters_per_class=1,

    n_samples=100, random_state=10

)



df = pd.DataFrame(X)

df['A64'] = y

df.A64.value_counts().plot(kind='bar', title='Count (target)');
from matplotlib import pyplot as plt

def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(

            X[y==l, 0],

            X[y==l, 1],

            c=c, label=l, marker=m

        )

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

pca = PCA(n_components=2)

X = pca.fit_transform(X)



plot_2d_space(X, y, 'Imbalanced dataset (2 PCA components)')
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(return_indices=True)

X_rus, y_rus, id_rus = rus.fit_sample(X, y)



print('Removed indexes:', id_rus)



plot_2d_space(X_rus, y_rus, 'Random under-sampling')
from imblearn.over_sampling import RandomOverSampler



ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X, y)



print(X_ros.shape[0] - X.shape[0], 'new random picked points')



plot_2d_space(X_ros, y_ros, 'Random over-sampling')
from imblearn.under_sampling import TomekLinks



tl = TomekLinks(return_indices=True, ratio='majority')

X_tl, y_tl, id_tl = tl.fit_sample(X, y)



print('Removed indexes:', id_tl)



plot_2d_space(X_tl, y_tl, 'Tomek links under-sampling')
from imblearn.under_sampling import ClusterCentroids



cc = ClusterCentroids(ratio={0: 10})

X_cc, y_cc = cc.fit_sample(X, y)



plot_2d_space(X_cc, y_cc, 'Cluster Centroids under-sampling')
from imblearn.over_sampling import SMOTE



smote = SMOTE(ratio='minority')

X_sm, y_sm = smote.fit_sample(X, y)



plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
from imblearn.combine import SMOTETomek



smt = SMOTETomek(ratio='auto')

X_smt, y_smt = smt.fit_sample(X, y)



plot_2d_space(X_smt, y_smt, 'SMOTE + Tomek links')