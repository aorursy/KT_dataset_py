import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from scipy.stats import zscore
myData = pd.read_csv('../input/vehicle.csv')

myData.head()
myData.shape
myData.isnull().sum()
myData.describe().transpose()
myData.info()
myData.fillna(myData.mean(), axis = 0, inplace = True)

myData.isnull().sum()

print(myData.shape)
myData.groupby('class').count()
plt.figure(figsize = (20,20))

sns.pairplot(data = myData, hue = 'class')
fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['compactness'], ax = g1)

g1.set_title('Distribution Plot')



sns.boxplot(myData['compactness'], ax = g2)

g2.set_title('Box Plot')
fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['circularity'], ax = g1)

g1.set_title('Distribution Plot')



sns.boxplot(myData['circularity'], ax = g2)

g2.set_title('Box Plot')
fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['distance_circularity'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['distance_circularity'], ax = g2)

g2.set_title("Box Plot")
fig, (g1, g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['radius_ratio'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['radius_ratio'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['radius_ratio'], 0.25)

q2 = np.quantile(myData['radius_ratio'], 0.50)

q3 = np.quantile(myData['radius_ratio'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("radius_ratio above ", myData['radius_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['radius_ratio'] > 276]['radius_ratio'].shape[0])
fig, (g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['pr.axis_aspect_ratio'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['pr.axis_aspect_ratio'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['pr.axis_aspect_ratio'], 0.25)

q2 = np.quantile(myData['pr.axis_aspect_ratio'], 0.50)

q3 = np.quantile(myData['pr.axis_aspect_ratio'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("pr.axis_aspect_ratio above ", myData['pr.axis_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['pr.axis_aspect_ratio'] > 77.0]['pr.axis_aspect_ratio'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['max.length_aspect_ratio'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['max.length_aspect_ratio'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['max.length_aspect_ratio'], 0.25)

q2 = np.quantile(myData['max.length_aspect_ratio'], 0.50)

q3 = np.quantile(myData['max.length_aspect_ratio'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("max.length_aspect_ratio above ", myData['max.length_aspect_ratio'].quantile(0.75) + (1.5*IQR), "are outliers")

print("max.length_aspect_ratio below ", myData['max.length_aspect_ratio'].quantile(0.25) - (1.5*IQR), "are outliers")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['scatter_ratio'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['scatter_ratio'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['elongatedness'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['elongatedness'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['pr.axis_rectangularity'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['pr.axis_rectangularity'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['max.length_rectangularity'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['max.length_rectangularity'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['scaled_variance'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['scaled_variance'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['scaled_variance'], 0.25)

q2 = np.quantile(myData['scaled_variance'], 0.50)

q3 = np.quantile(myData['scaled_variance'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("scaled_variance above ", myData['scaled_variance'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['scaled_variance'] > 292]['scaled_variance'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['scaled_variance.1'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['scaled_variance.1'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['scaled_variance.1'], 0.25)

q2 = np.quantile(myData['scaled_variance.1'], 0.50)

q3 = np.quantile(myData['scaled_variance.1'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("scaled variance.1 above ", myData['scaled_variance.1'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['scaled_variance.1'] > 989.5]['scaled_variance.1'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['scaled_radius_of_gyration'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['scaled_radius_of_gyration'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['scaled_radius_of_gyration.1'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['scaled_radius_of_gyration.1'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.25)

q2 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.50)

q3 = np.quantile(myData['scaled_radius_of_gyration.1'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("scaled radius of gyration.1 above ", myData['scaled_radius_of_gyration.1'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['scaled_radius_of_gyration.1'] > 87]['scaled_radius_of_gyration.1'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['skewness_about'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['skewness_about'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['skewness_about'], 0.25)

q2 = np.quantile(myData['skewness_about'], 0.50)

q3 = np.quantile(myData['skewness_about'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("skewness about above ", myData['skewness_about'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['skewness_about'] > 19.5]['skewness_about'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['skewness_about.1'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['skewness_about.1'], ax = g2)

g2.set_title("Box Plot")
q1 = np.quantile(myData['skewness_about.1'], 0.25)

q2 = np.quantile(myData['skewness_about.1'], 0.50)

q3 = np.quantile(myData['skewness_about.1'], 0.75)

IQR = q3 - q1



print("Quartile q1: ", q1)

print("Quartile q2: ", q2)

print("Quartile q3: ", q3)

print("Inter Quartile Range: ", IQR)



print("skewness about.1 above ", myData['skewness_about.1'].quantile(0.75) + (1.5*IQR), "are outliers")

print("No. of outliers ", myData[myData['skewness_about.1'] > 40]['skewness_about.1'].shape[0])
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['skewness_about.2'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['skewness_about.2'], ax = g2)

g2.set_title("Box Plot")
fig,(g1,g2) = plt.subplots(nrows = 1, ncols = 2)

fig.set_size_inches(15,2)

sns.distplot(myData['hollows_ratio'], ax = g1)

g1.set_title("Distribution Plot")



sns.boxplot(myData['hollows_ratio'], ax = g2)

g2.set_title("Box Plot")
myData.groupby('class').count()
sns.countplot(myData['class'])
plt.figure(figsize = (20,10))

sns.heatmap(myData.corr(), annot = True)
myData_attr = myData.drop('class', axis = 1)

myData_target = myData['class']



print(myData_attr.shape)

print(myData_target.shape)
myData_attr_s = myData_attr.apply(zscore)
myData_target.replace({"car": 0, "bus": 1, "van": 2}, inplace = True)

print(myData_target.shape)
cov_mat = np.cov(myData_attr_s, rowvar = False)

print(cov_mat)
print(cov_mat.shape)
from sklearn.decomposition import PCA

pca_18 = PCA(n_components = 18)

pca_18.fit(myData_attr_s)
print(pca_18.explained_variance_)
print(pca_18.components_)
print(pca_18.explained_variance_ratio_)
plt.bar(list(range(1,19)), pca_18.explained_variance_ratio_, alpha = 0.5)

plt.ylabel('Variation explained')

plt.xlabel('Eigen values')

plt.show()
plt.step(list(range(1,19)), np.cumsum(pca_18.explained_variance_ratio_), where = 'mid')

plt.ylabel('Cum of variation explained')

plt.xlabel('Eigen value')

plt.show()
pca_8 = PCA(n_components = 8)

pca_8.fit(myData_attr_s)

print(pca_8.components_)

print(pca_8.explained_variance_ratio_)
myData_attr_s_pca_8 = pca_8.transform(myData_attr_s)

myData_attr_s_pca_8.shape
sns.pairplot(pd.DataFrame(myData_attr_s_pca_8))
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

from sklearn.svm import SVC
accuracies = {}

model = SVC()



X_train, X_test, y_train, y_test = train_test_split(myData_attr_s_pca_8, myData_target, test_size = 0.30, random_state = 1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_svm = model.score(X_test, y_test) *100



accuracies['SVM'] = acc_svm

print(model.score(X_train, y_train))

print(model.score(X_test, y_test))
print(classification_report(y_test,y_pred))
param = {

    'C' : [0.01,0.05,0.5,1],

    'kernel' :['linear','rbf']

}



grid_svm = GridSearchCV(model, param_grid = param, scoring = 'accuracy', cv = 10)
grid_svm.fit(X_train,y_train)
grid_svm.best_params_
model_svm = SVC(C = 1, kernel = 'rbf', gamma = 1)

X_train, X_test, y_train, y_test = train_test_split(myData_attr_s_pca_8, myData_target, test_size = 0.30, random_state = 1)

model_svm.fit(X_train, y_train)

y_pred = model_svm.predict(X_test)



acc_svm_gs = model_svm.score(X_test, y_test) * 100

accuracies['SVM_GS'] = acc_svm_gs

print(model.score(X_test, y_test))

print(classification_report(y_test, y_pred))
svm_eval = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)

svm_eval.mean()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(myData_attr_s, myData_target)

rf.score(myData_attr_s, myData_target)



feature_importances = pd.DataFrame(rf.feature_importances_, index = myData_attr_s.columns,

                                  columns = ['importance']).sort_values('importance', ascending = False) * 100



feature_importances
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics





nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

expected = y_test

predicted = nb_model.predict(X_test)



acc_nb = nb_model.score(X_test, y_test) * 100

accuracies['NB'] = acc_nb



print(metrics.classification_report(expected, predicted))

print('Total accuracy: ', np.round(metrics.accuracy_score(expected, predicted), 2))
plt.figure(figsize = (8,5))

plt.yticks(np.arange(0,100,10))

sns.barplot(x = list(accuracies.keys()), y = list(accuracies.values()))
models = pd.DataFrame({

    'Model': ['SVM', 'SVM_GS','Naive Bayes'],

    

    'Score': [acc_svm, acc_svm_gs, acc_nb]

    })



models.sort_values(by='Score', ascending=False)
y_cm_svm = model.predict(X_test)

y_cm_svm_gs = model_svm.predict(X_test)

y_cm_nb = nb_model.predict(X_test)



from sklearn.metrics import confusion_matrix



cm_svm = confusion_matrix(y_test, y_cm_svm)

cm_svm_gs = confusion_matrix(y_test, y_cm_svm_gs)

cm_nb = confusion_matrix(y_test, y_cm_nb)



plt.figure(figsize = (16,4))

plt.suptitle("Confusion Matrices",fontsize=12)

plt.subplots_adjust(wspace = 0.8, hspace = 0.8)



plt.subplot(1,3,1)

plt.title("SVM Confusion Matrix")

sns.heatmap(cm_svm, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})





plt.subplot(1,3,2)

plt.title("SVM Grid Search Confusion Matrix")

sns.heatmap(cm_svm_gs, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(1,3,3)

plt.title("NB Confusion Matrix")

sns.heatmap(cm_nb, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})