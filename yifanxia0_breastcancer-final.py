import pandas as pd

import numpy as np

import math

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

from sklearn import preprocessing

from sklearn.decomposition import PCA 

from sklearn.manifold import TSNE

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import NearestNeighbors

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn import tree



from xgboost import XGBClassifier



% matplotlib inline



# Set Random Seed

np.random.seed(42)

np.random.RandomState(42)
# Read csv

data = pd.read_csv("../input/data 2.csv")

data.head()
print ("Number of data points :", len(data))
# Describe data

data.describe()
data.info()
# Save labels in y

y = data["diagnosis"]
# Drop columns

X = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
y_pos = [yy for yy in y if yy == 'M']

y_neg = [yy for yy in y if yy == 'B']

print('y_pos: ', len(y_pos))

print('y_neg: ', len(y_neg))
# Plot a Correlation chart

corr = X.corr() # .corr is used for find corelation

#plt.figure(figsize=(20,15))

sns.set(rc={'figure.figsize':(25,20)})

# plot a heatmap

sns.heatmap(corr, cbar = True, annot=True, fmt= '.2f',annot_kws={'size': 10},

           xticklabels= X.columns, yticklabels= X.columns,

           cmap= 'viridis') 
# Drop columns

X = X.drop(["perimeter_mean", "area_mean", "radius_worst", "area_worst", "perimeter_worst"], axis=1)

X = X.drop(["texture_worst", "perimeter_se", "area_se"], axis=1)



X.info()
# Plot a countplot

sns.set(rc={'figure.figsize':(8,5)})

sns.countplot(y) 
# Print count

count = y.value_counts()

print('Number of Benign : ',count[0] )

print('Number of Malignant : ',count[1]) 
# Creating a empty list

mean_volume = []

# defining pi

pi = 3.1415



# calculatin mean volume for each mean radius and saving result in mean_volume list

for i in range(len(X)):

    #aving result in mean_volume list

    mean_volume.append((math.pow(X["radius_mean"][i], 3)*4*pi)/3)



# Creating a new feature

X["mean_volume"]= mean_volume    
# Creating a new feature adding up some phisical measuraments

# X["mesuraments_sum_mean"] = X["radius_mean"] + X["perimeter_mean"] + X["area_mean"]
X.head()
# Define a scaler function

def scaler(df):

    """The Function receive a Dataframe and return a Scaled Dataframe"""

    scaler = preprocessing.MinMaxScaler()

    scaled_df = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    

    return scaled_df



# testing scaler

scaled_df = scaler(X)



scaled_df.head()
# Define a function to detect outliers

def remove_outliers(X, y, f=2, distance=1.5):

    

    """The Function receive Features (X) and Label (y) a frequency (f) and Inter-Quartile distance (distance),  

    and return features and labels without outliers (good_X, good_y)"""

    

    outliers  = []



    # For each feature find the data points with extreme high or low values

    for feature in X.keys():



        # Calculate Q1 (25th percentile of the data) for the given feature

        Q1 = np.percentile(X[feature], 25)



        # Calculate Q3 (75th percentile of the data) for the given feature

        Q3 = np.percentile(X[feature], 75)



        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

        step = (Q3 - Q1) * distance



        outliers.append(X[~((X[feature] >= Q1 - step) & (X[feature] <= Q3 + step))].index.values)



    # Select the indices for data points you wish to remove

    flat_list = [item for sublist in outliers for item in sublist]



    # importing Counter

    from collections import Counter

    

    freq = Counter(flat_list)

    # Create a list to store outliers to remove

    outliers_to_remove = []

    

    for key, value in freq.items():

        if value > f:

            outliers_to_remove.append(key)



    # Remove the outliers, if any were specified

    good_X = X.drop(X.index[outliers_to_remove]).reset_index(drop = True)

    good_y    = y.drop(y.index[outliers_to_remove]).reset_index(drop = True)

    # Sort list

    outliers_to_remove.sort()

    # Print outliers founded

    for i in range(len(outliers_to_remove)):

        print( "data point: ", outliers_to_remove[i], "is considered outlier to more than ", f, " feature" )



    print( "All ", len(outliers_to_remove), "were removed!" )

    # return data without outliers

    return good_X, good_y 





good_X, good_y = remove_outliers(scaled_df, y, f=2, distance=1.5)
good_X.head()
sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(good_y) 
count = y.value_counts()

count2 = good_y.value_counts()



print('Number of Benign removed: ',count[0] - count2[0])

print('Number of Malignant removed: ',count[1] - count2[1])
def tsne_plot(good_X, good_y):

    tsne = TSNE(n_components=2, init='random', random_state=0, perplexity=100)

    X_original_SNE = tsne.fit_transform(good_X)



    X_original_SNE_df = pd.DataFrame(X_original_SNE, columns=["d1", "d2"])

    good_y_df = pd.DataFrame(good_y, columns=["diagnosis"])



    X_original_SNE_df = pd.concat([good_y_df, X_original_SNE_df.iloc[:,0:]],axis=1)

    X_original_SNE_df.head()



    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(111)



    m_SNE = X_original_SNE_df.loc[X_original_SNE_df['diagnosis'] == 'M']

    b_SNE = X_original_SNE_df.loc[X_original_SNE_df['diagnosis'] == 'B']



    ax.scatter(m_SNE['d1'], m_SNE['d2'], c='darkorange', s=100)

    ax.scatter(b_SNE['d1'], b_SNE['d2'], c='blue', s=100)

    

    ax.set_xlim([-10, 10])

    ax.set_ylim([-10, 10])



    plt.show()
tsne_plot(scaled_df, y)
tsne_plot(good_X, good_y)
X_resampled, y_resampled = ADASYN().fit_sample(good_X, good_y)

print(sorted(Counter(y_resampled).items()))



sns.set(rc={'figure.figsize':(8,5)})



sns.countplot(y_resampled) 
tsne_plot(X_resampled, y_resampled)
def selector(X, y, k=12):

    

    """The function receive features and labels (X, y) and a target number to select features (k)

    and return a new dataset wiht k best features"""

    

    X = pd.DataFrame(X)

    

    selector = SelectKBest(chi2, k)

    

    X_new = selector.fit_transform(X, y)

    

    return pd.DataFrame(X_new, columns=X.columns[selector.get_support()])



X_select = selector(X_resampled, y_resampled)



X_select.head()
# Support Vector Machine Classifier

SV_clf = svm.SVC()

# Parameters to tune

SV_par = {'kernel': ['rbf'], 'C': [1]}



# Logistic Regression

LR_clf = LogisticRegression()

# Parameters to tune

LR_par= {'penalty':['l1','l2'], 'C': [0.5, 1, 5, 10], 'max_iter':[50, 100, 150, 200]}



classifiers = [SV_clf, LR_clf]



classifiers_names = ['Support Vector Machine', 'Logistic Regression']



parameters = [SV_par, LR_par]
def tune_compare_clf(X, y, classifiers, parameters, classifiers_names):

    

    '''The function receive Data (X, y), a classifiers list, 

    a list of parameters to tune each chassifier (each one is a dictionary), 

    and a list with classifiers name. 

    

    The function split data in Train and Test data, 

    train and tune all algorithms and print results using F1 score.

    

    The function also returns a Dataframe with predictions, each row is a classifier prediction,

    and X_test and y_test.

    '''

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    



    print("\n" "Train size : ", X_train.shape, " and Train labels : ", y_train.shape, "\n")



    print("Test size: ", X_test.shape, " and Test labels : ", y_test.shape, "\n", "\n")

    

    results = []

    

    print("  ---- F1 Score  ----  ", "\n")



    for clf, par, name in zip(classifiers, parameters, classifiers_names):

        # Store results in results list

        clf_tuned = GridSearchCV(clf, par).fit(X_train, y_train)

        y_pred = clf_tuned.predict(X_test)

        results.append(y_pred)   



        print(name, ": %.2f%%" % (f1_score(y_test, y_pred, average='weighted') * 100.0))



    result = pd.DataFrame.from_records(results)   

    

    return result, X_test,  y_test
# result, X_test, y_test = tune_compare_clf(X_new, y_new, classifiers, parameters, classifiers_names)

# result, X_test, y_test = tune_compare_clf(X_resampled, y_resampled, classifiers, parameters, classifiers_names)
# result, X_test, y_test = tune_compare_clf(X_select, y_resampled, classifiers, parameters, classifiers_names)
from sklearn.neighbors import KNeighborsClassifier  

from sklearn import datasets

from sklearn import metrics

from sklearn.model_selection import train_test_split



def mesh(f1, f2, expend):

    x_min, x_max = f1.min() - expend, f1.max() + expend

    y_min, y_max = f2.min() - expend, f2.max() + expend

    resolution = 80

    hx = (x_max - x_min)/resolution

    hy = (y_max - y_min)/resolution

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    return xx, yy



def knn(X, y):

#     X = X.values

#     y = y.values

    

    acc = 0

    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True) # 70% training and 30% test



        # we create an instance of Neighbours Classifier and fit the data.

        clf = KNeighborsClassifier(n_neighbors=29)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)



#         print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

        acc += metrics.accuracy_score(y_test, y_pred)

    print("Accuracy:", acc/100)



knn(X_resampled, y_resampled)
knn(X_select, y_resampled)
from sklearn import svm



def svm_fit(X, y):    

    acc = 0

    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True) # 70% training and 30% test



        clf = svm.SVC(gamma='scale')

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

    

        acc += metrics.accuracy_score(y_test, y_pred)

        

    print("Accuracy:", acc/100)



svm_fit(X_resampled, y_resampled)
svm_fit(X_select, y_resampled)
import numpy as np

import math

import matplotlib.pyplot as plt

import pandas as pd

import operator 

import csv

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split



def euclideanDistance(instance1, instance2, length):

    distance = 0

    for x in range(length):

        distance += pow((instance1[x] - instance2[x]), 2)

    return math.sqrt(distance)



def getNeighbors(trainingSet, testInstance, k):

    distances = []

    length = len(testInstance)

    for x in range(len(trainingSet)):

        dist = euclideanDistance(testInstance, trainingSet[x], length)

        distances.append((trainingSet[x], dist))

    distances.sort(key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):

        neighbors.append(distances[x][0])

    return neighbors



def count(df, threshold):

    df = pd.DataFrame(df)

    data_0 = df.loc[df[0] == 0]

    data_1 = df.loc[df[0] == 1]



    data_0_strip = data_0.loc[data_0[1] > threshold]

    data_1_strip = data_1.loc[data_1[1] > threshold]



    pfa = len(data_0_strip) / len(data_0)

    pde = len(data_1_strip) / len(data_1)



    return pfa, pde



def fit(thresholds, data):

    fpr, tpr = [], []

    pfa, pde = count(data, -math.inf)

    fpr.append(pfa)

    tpr.append(pde)



    for thres in thresholds:

        pfa, pde = count(data, thres)

        fpr.append(pfa)

        tpr.append(pde)



    pfa, pde = count(data, math.inf)

    fpr.append(pfa)

    tpr.append(pde)



    return fpr, tpr



def operatingpoint(ph0, ph1, pfa, pde):

    max_pcd = 0

    for idx, pfaa in enumerate(pfa):

        pcd = ph0 * (1 - pfaa) + ph1 * (pde[idx])

        if pcd > max_pcd:

            max_pcd = pcd

            point = (pfaa, pde[idx])

    return point, max_pcd



def predict(train_point_list, target, label, k):

    dist_list = np.array([np.linalg.norm(p - target) for p in train_point_list])

    ind = np.argpartition(dist_list, k)[:k]

    return(sum(label[ind])/k)



def roc_curve(label, preds):

    beta = np.unique(preds)

    beta = np.sort(beta)

    beta = np.insert(beta, 0, -np.inf)

    beta = np.insert(beta, beta.size, np.inf)

    H0 = preds[label == 0]

    H1 = preds[label == 1]

    pfa = np.array([sum(H0 > b)/H0.size for b in beta])

    pd = np.array([sum(H1 > b)/H1.size for b in beta])

    return pfa, pd



def max_Pcd(pfa, pd, label):

    ph1 = sum(label == 1)/len(label)

    ph0 = 1-ph1

    pcd = pd*ph1-ph0*pfa+ph0

    return np.max(pcd)



def calpe(df, df_label, testing, testing_label, K):

    roc = []

    Z = np.array([predict(df, pt, df_label, K) for pt in testing])

    pfa, pd = roc_curve(testing_label, Z)

    return 1 - max_Pcd(pfa, pd, testing_label)
def knn_pe(df, df_label, testing, testing_label):

    trainPE, testPE = [], []

    xaxisTrain, xaxisTest = [], []

    for K in range(1, 400, 4):

        minpeTraining = calpe(df, df_label, df, df_label, K)

        minpeTesting = calpe(df, df_label, testing, testing_label, K)



        trainPE.append(minpeTraining)

        xaxisTrain.append(len(df) / K)

        

        testPE.append(minpeTesting)

        xaxisTest.append(len(testing) / K)



    # PLOTTING

    plt.title('min Pe with N/K Testing on Training Data')

    plt.xlabel('N / K', ha='center', va='center')

    plt.ylabel('min Pe', ha='center', va='center', rotation='vertical')



    plt.plot(xaxisTrain, trainPE, color='darkorange', lw=3, label='Training Data')

    plt.plot(xaxisTest, testPE, color='g', lw=3, label='Testing Data')



    plt.legend(loc="upper right")

    plt.grid()



    plt.show()

    

y_label = []

for x in y_resampled:

    if x == 'M':

        y_label.append(1)

    else:

        y_label.append(0)

y_label = np.array(y_label)



X_resampled_train, X_resampled_test, y_label_train, y_label_test = train_test_split(X_resampled, y_label, test_size=0.15, random_state=42, shuffle=True)



knn_pe(X_resampled_train, y_label_train, X_resampled_test, y_label_test)
# Scale, Outliers Remove and Resample

    

X_scaled = scaler(X)

X_good, y_good = remove_outliers(X_scaled, y, f=2, distance=2)

X_new, y_new = resample(X_good, y_good, method="RandomOverSampler")



result, X_test, y_test = tune_compare_clf(X_new, y_new, classifiers, parameters, classifiers_names)
y_pred_votes = result.describe().iloc[[2]]
print("Accuracy: %.2f%%" % (f1_score(y_test, y_pred_votes.T, average='weighted') * 100.0))



sns.set(rc={'figure.figsize':(5,5)})

cm = confusion_matrix(y_test,y_pred_votes.T)

sns.heatmap(cm,annot=True,fmt="d")