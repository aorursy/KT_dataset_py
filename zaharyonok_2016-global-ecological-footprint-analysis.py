# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ecological_footprint = pd.read_csv("/kaggle/input/ecological-footprint/countries.csv")

#ecological_footprint.dropna(how = 'any', inplace=True)
print(ecological_footprint.columns)

print(ecological_footprint.dtypes)

print(ecological_footprint.shape)
# Relationship between GDP (Gross domestic product) and HDI(Human Development Index)

HDI_GDP = pd.DataFrame(ecological_footprint, columns=['Country','HDI', 'GDP per Capita'])

print('Before cleaning', HDI_GDP.isnull().sum())

HDI_GDP.dropna(how='any', inplace=True)

print('After cleaning', HDI_GDP.isnull().sum())

GDP = HDI_GDP['GDP per Capita']

GDP = GDP.replace({'\$': '', ',': ''}, regex=True).astype(float)

HDI_GDP['GDP per Capita'] = GDP

print(HDI_GDP.dtypes)

HDI_GDP.plot(kind='scatter', x='GDP per Capita', y='HDI')
# Relationship between GDP (Gross domestic product) and Total Ecological Footprint

GDP_totalfootprint = pd.DataFrame(ecological_footprint, columns=['Country', 'GDP per Capita', 'Total Ecological Footprint'])

print('Before cleaning', GDP_totalfootprint.isnull().sum())

GDP_totalfootprint.dropna(how='any', inplace=True)

print('After cleaning', GDP_totalfootprint.isnull().sum())

GDP = GDP_totalfootprint['GDP per Capita']

GDP = GDP.replace({'\$': '', ',': ''}, regex=True).astype(float)

GDP_totalfootprint['GDP per Capita'] = GDP

GDP_totalfootprint.plot(kind='scatter', x='GDP per Capita', y='Total Ecological Footprint')
# HDI(Human Development Index) and Total Ecological Footprint

HDI_totalfootprint = pd.DataFrame(ecological_footprint, columns=['Country', 'HDI', 'Total Ecological Footprint'])

print('Before cleaning', HDI_totalfootprint.isnull().sum())

HDI_totalfootprint.dropna(how='any', inplace=True)

print('After cleaning', HDI_totalfootprint.isnull().sum())

HDI_totalfootprint.plot(kind='scatter', x='HDI', y='Total Ecological Footprint')





highest_HDI = HDI_GDP[HDI_GDP['HDI']>0.8]

print(highest_HDI.describe())

print(highest_HDI.sort_values('HDI'))
HDI_GDP_Eco_Footprint = pd.DataFrame(ecological_footprint, columns=['Country','HDI', 'GDP per Capita', 'Total Ecological Footprint'])

HDI_GDP_Eco_Footprint.sort_values('Total Ecological Footprint')
print(ecological_footprint.head())
import seaborn as sns

import matplotlib.pyplot as plt

GDP = ecological_footprint['GDP per Capita']

GDP = GDP.replace({'\$': '', ',': ''}, regex=True).astype(float)

ecological_footprint['GDP per Capita'] = GDP

sns.pairplot(ecological_footprint)

plt.show()
corr = ecological_footprint.corr()

# show the strenght of relationship

corr.style.background_gradient(cmap='coolwarm')
ecological_footprint.isnull().sum()
ecological_footprint['HDI'].fillna(0, inplace=True)

ecological_footprint['GDP per Capita'].fillna(0, inplace=True)

ecological_footprint['Cropland Footprint'].fillna(0, inplace=True)

ecological_footprint['Grazing Footprint'].fillna(0, inplace=True)

ecological_footprint['Forest Footprint'].fillna(0, inplace=True)

ecological_footprint['Carbon Footprint'].fillna(0, inplace=True)

ecological_footprint['Fish Footprint'].fillna(0, inplace=True)

ecological_footprint['Cropland'].fillna(0, inplace=True)

ecological_footprint['Grazing Land'].fillna(0, inplace=True)

ecological_footprint['Forest Land'].fillna(0, inplace=True)

ecological_footprint['Fishing Water'].fillna(0, inplace=True)

ecological_footprint['Urban Land'].fillna(0, inplace=True)

ecological_footprint.isnull().sum()

from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()

from sklearn.metrics import mean_squared_error

#x

population = ecological_footprint['Population (millions)']

population = population.apply(lambda x:x/100)

HDI = ecological_footprint['HDI']

GDP = ecological_footprint['GDP per Capita']

crop_fft = ecological_footprint['Cropland Footprint']

grzng_fft = ecological_footprint['Grazing Footprint']

forest_fft = ecological_footprint['Forest Footprint']

carbon_fft = ecological_footprint['Carbon Footprint']

fish_fft = ecological_footprint['Fish Footprint']

crop_lnd = ecological_footprint['Cropland']

grz_lnd = ecological_footprint['Grazing Land']

forest_lnd = ecological_footprint['Forest Land']

fish_water = ecological_footprint['Fishing Water']

urb_lnd = ecological_footprint['Urban Land']

ttl_biocapacity = ecological_footprint['Total Biocapacity']

#y

total_footprint = ecological_footprint['Total Ecological Footprint']

x = np.array([population,HDI,GDP,crop_fft,grzng_fft,forest_fft,forest_fft,carbon_fft,fish_fft,crop_lnd,grz_lnd,forest_lnd,fish_water,urb_lnd,ttl_biocapacity]).T

y = np.array([total_footprint]).T

x_train, x_test, y_train, y_test = train_test_split(x, \

                                                    total_footprint, \

                                                    test_size=0.2, \

                                                    random_state=4)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE ", mse)

print("MSE ",mean_squared_error(y_test, y_pred))
from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()

from sklearn.metrics import mean_squared_error

#x

population = ecological_footprint['Population (millions)']

population = population.apply(lambda x:x/100)

HDI = ecological_footprint['HDI']

GDP = ecological_footprint['GDP per Capita']

fish = ecological_footprint['Fish Footprint']

grzng_land = ecological_footprint['Grazing Land']

forest = ecological_footprint['Forest Land']

fish_water = ecological_footprint['Fishing Water']

ttl_biocapacity = ecological_footprint['Total Biocapacity']

#y

total_footprint = ecological_footprint['Total Ecological Footprint']

x = np.array([population,HDI,GDP,fish,grzng_land,forest,fish_water,ttl_biocapacity]).T

y = np.array([total_footprint]).T

x_train, x_test, y_train, y_test = train_test_split(x, \

                                                    total_footprint, \

                                                    test_size=0.2, \

                                                    random_state=4)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE ", mse)

print("MSE ",mean_squared_error(y_test, y_pred))
cor_target = abs(corr['Total Ecological Footprint'])#Selecting least correlated features

relevant_features = cor_target[cor_target<0.2]

relevant_features
from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()

from sklearn.metrics import mean_squared_error

#x

population = ecological_footprint['Population (millions)']

population = population.apply(lambda x:x/100)

fish = ecological_footprint['Fish Footprint']

grzng_land = ecological_footprint['Grazing Land']

forest = ecological_footprint['Forest Land']

fish_water = ecological_footprint['Fishing Water']

ttl_biocapacity = ecological_footprint['Total Biocapacity']

#y

total_footprint = ecological_footprint['Total Ecological Footprint']

x = np.array([population,fish,grzng_land,forest,fish_water,ttl_biocapacity]).T

y = np.array([total_footprint]).T

x_train, x_test, y_train, y_test = train_test_split(x, \

                                                    total_footprint, \

                                                    test_size=0.2, \

                                                    random_state=4)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE ", mse)

print("MSE ",mean_squared_error(y_test, y_pred))
cor_target = abs(corr['Total Ecological Footprint'])#Selecting least correlated features

relevant_features = cor_target[cor_target<0.1]

relevant_features
from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()

from sklearn.metrics import mean_squared_error

#x

population = ecological_footprint['Population (millions)']

population = population.apply(lambda x:x/100)

grzng_land = ecological_footprint['Grazing Land']

forest = ecological_footprint['Forest Land']

ttl_biocapacity = ecological_footprint['Total Biocapacity']

#y

total_footprint = ecological_footprint['Total Ecological Footprint']

x = np.array([population,grzng_land,forest,ttl_biocapacity]).T

y = np.array([total_footprint]).T

x_train, x_test, y_train, y_test = train_test_split(x, \

                                                    total_footprint, \

                                                    test_size=0.2, \

                                                    random_state=4)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE ", mse)

print("MSE ",mean_squared_error(y_test, y_pred))
cor_target = abs(corr['Total Ecological Footprint'])#Selecting least correlated features

relevant_features = cor_target[cor_target<0.06]

relevant_features
from sklearn import linear_model

from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()

from sklearn.metrics import mean_squared_error

#x

population = ecological_footprint['Population (millions)']

population = population.apply(lambda x:x/100)

forest = ecological_footprint['Forest Land']

#y

total_footprint = ecological_footprint['Total Ecological Footprint']

x = np.array([population, forest]).T

y = np.array([total_footprint]).T

x_train, x_test, y_train, y_test = train_test_split(x, \

                                                    total_footprint, \

                                                    test_size=0.2, \

                                                    random_state=4)

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)

mse = np.mean((y_pred-y_test)**2)

print("MSE ", mse)

print("MSE ",mean_squared_error(y_test, y_pred))

print(ecological_footprint.shape)
print(ecological_footprint['Total Ecological Footprint'])
total_footprint = ecological_footprint['Total Ecological Footprint']

temp_arr=[]

for impact in total_footprint:

    if impact >= 10:

        temp_arr.append(2)

    if impact < 5:

        temp_arr.append(0)

    else:

        temp_arr.append(1)

#print(temp_arr)

total_footprint_categorized = total_footprint.copy()

i = 0

while i < 188:

    total_footprint_categorized[i] = temp_arr[i]

    i = i+1

#print(total_footprint_categorized[0:50])

print(total_footprint_categorized.shape)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import numpy as np

scaler = StandardScaler()

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



x = ecological_footprint[["Population (millions)","HDI","GDP per Capita","Cropland Footprint","Grazing Footprint","Forest Footprint","Carbon Footprint","Fish Footprint",

"Cropland","Grazing Land","Forest Land","Fishing Water","Urban Land","Total Biocapacity"]].copy(deep=True)

y = total_footprint_categorized

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

model = LogisticRegression(penalty='l2', C=1)

y_train = np.array(y_train).flatten()

model.fit(x_train, y_train)

ypred = model.predict(x_test)

confusion_matrix(y_test, ypred)

print("Accuracy rate is: %0.2f" %(accuracy_score(y_test, ypred)))

logit_roc_auc = roc_auc_score(y_test, ypred)

print("Logistic Area under the curve = %0.2f" %logit_roc_auc)

print(classification_report(y_test, ypred))



b = model.predict_proba(x_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, b)

plt.figure()

plt.plot(fpr, tpr, label='ROC(Receiver Operator Characteristic Curve) curve (area = %0.2f)' %logit_roc_auc)

plt.plot([0,1], [0,1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
x = ecological_footprint[["Population (millions)","Forest Footprint","Urban Land","Total Biocapacity"]].copy(deep=True)

y = total_footprint_categorized

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

model = LogisticRegression(penalty='l2', C=1)

y_train = np.array(y_train).flatten()

model.fit(x_train, y_train)

ypred = model.predict(x_test)

confusion_matrix(y_test, ypred)

print("Accuracy rate is: %0.2f" %(accuracy_score(y_test, ypred)))

logit_roc_auc = roc_auc_score(y_test, ypred)

print("Logistic Area under the curve = %0.2f" %logit_roc_auc)

print(classification_report(y_test, ypred))



b = model.predict_proba(x_test)[:,1]

fpr, tpr, threshold = roc_curve(y_test, b)

plt.figure()

plt.plot(fpr, tpr, label='ROC(Receiver Operator Characteristic Curve) curve (area = %0.2f)' %logit_roc_auc)

plt.plot([0,1], [0,1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
x = ecological_footprint[["Population (millions)","HDI","GDP per Capita","Cropland Footprint","Grazing Footprint","Forest Footprint","Carbon Footprint","Fish Footprint",

"Cropland","Grazing Land","Forest Land","Fishing Water","Urban Land","Total Biocapacity"]].copy(deep=True)

y = total_footprint_categorized

x = scaler.fit_transform(x)

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x, \

                                                    y, \

                                                    test_size=0.2, \

                                                    random_state=4)

from sklearn.svm import SVC

svclassifier = SVC(kernel='poly', degree=4)

svclassifier.fit(x_train_svm, y_train_svm)
y_pred_svm = svclassifier.predict(x_test_svm)

from sklearn.metrics import classification_report, confusion_matrix

print(accuracy_score(y_test_svm, y_pred_svm))



print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))

svclassifier = SVC(kernel='rbf')

svclassifier.fit(x_train_svm, y_train_svm)
y_pred_svm = svclassifier.predict(x_test_svm)

print(accuracy_score(y_test_svm, y_pred_svm))



print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))

svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(x_train_svm, y_train_svm)



y_pred_svm = svclassifier.predict(x_test_svm)

print(accuracy_score(y_test_svm, y_pred_svm))



print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))
x = ecological_footprint[["Population (millions)","Forest Footprint","Urban Land","Total Biocapacity"]].copy(deep=True)

y = total_footprint_categorized

x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
svclassifier = SVC(kernel='poly', degree=4)

svclassifier.fit(x_train_svm, y_train_svm)

y_pred_svm = svclassifier.predict(x_test_svm)

print(accuracy_score(y_test_svm, y_pred_svm))



print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))
svclassifier = SVC(kernel='rbf')

svclassifier.fit(x_train_svm, y_train_svm)

y_pred_svm = svclassifier.predict(x_test_svm)



print(accuracy_score(y_test_svm, y_pred_svm))

print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))
vclassifier = SVC(kernel='sigmoid')

svclassifier.fit(x_train_svm, y_train_svm)



y_pred_svm = svclassifier.predict(x_test_svm)

print(accuracy_score(y_test_svm, y_pred_svm))



print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))
print(ecological_footprint.describe(percentiles=[0.25,0.5,0.75]))
bins = [0,1.4825,2.74,4.64,15.82]

group_names = [0,1,2,3]
ecological_footprint['eco_foot_class'] = pd.cut(ecological_footprint['Total Ecological Footprint'], bins, labels=group_names)
print(ecological_footprint['eco_foot_class'])
# We do SVM first on the model with all the features---only going to use rbf for brevity

x = ecological_footprint[["Population (millions)","HDI","GDP per Capita","Cropland Footprint","Grazing Footprint","Forest Footprint","Carbon Footprint","Fish Footprint",

"Cropland","Grazing Land","Forest Land","Fishing Water","Urban Land","Total Biocapacity"]].copy(deep=True)

y = ecological_footprint['eco_foot_class']

x = scaler.fit_transform(x)

x_train_svm, x_test_svm, y_train_svm, y_test_svm = train_test_split(x, \

                                                    y, \

                                                    test_size=0.2, \

                                                    random_state=4)



svclassifier = SVC(kernel='rbf', degree=4)

svclassifier.fit(x_train_svm, y_train_svm)

y_pred_svm = svclassifier.predict(x_test_svm)

print(confusion_matrix(y_test_svm, y_pred_svm))

print(classification_report(y_test_svm, y_pred_svm))
# Decision Tree using the percentile classifier

from sklearn import tree



tree_model = tree.DecisionTreeClassifier()

tree_model.fit(x_train_svm, y_train_svm)

y_predict_tree = tree_model.predict(x_test_svm)



from sklearn.metrics import accuracy_score



accuracy_score(y_test_svm, y_predict_tree)



print(confusion_matrix(y_test_svm, y_predict_tree))

print(classification_report(y_test_svm, y_predict_tree))

#tree.export_graphviz(tree_model.tree_, out_file='tree.dot', feature_names=ecological_footprint.columns)

# could not get graphviz to work
# Now doing some graphs to color by region and marker size based on population

# The legends using legend_elements don't work here--examples online are the same

ef = pd.read_csv("/kaggle/input/ecological-footprint/countries.csv")



reg = {'Middle East/Central Asia':0,'Northern/Eastern Europe':1,'Africa':2,'Latin America':3,'Asia-Pacific':4,'European Union':5,'North America':6}

#print(ef)

eff = ef.copy(deep=True)

eff['Region'] = ef['Region'].replace(reg,inplace=True)



#print(ef.head())

#print(ef.dtypes)

fig,ax = plt.subplots()

fig.suptitle('Total Ecological Footprint vs HDI by Country/Region')

scatter = ax.scatter(x=ef['HDI'],y=ef['Total Ecological Footprint'],c=ef['Region'],s=ef['Population (millions)'])



plt.xlabel('HDI')

plt.ylabel('Total Ecological Footprint')

fig.suptitle('Total Ecological Footprint vs HDI by Country/Region')
fig, axs = plt.subplots(2,3,figsize=(10,10))



#fig.suptitle('Vertically stacked subplots')

axs[0,0].scatter(x=ef['HDI'],y=ef['Carbon Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[0,0].set_title('Carbon Footprint')

axs[0,1].scatter(x=ef['HDI'],y=ef['Cropland Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[0,1].set_title('Cropland Footprint')

axs[0,2].scatter(x=ef['HDI'],y=ef['Fish Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[0,2].set_title('Fish Footprint')

axs[1,0].scatter(x=ef['HDI'],y=ef['Grazing Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[1,0].set_title('Grazing Footprint')

axs[1,1].scatter(x=ef['HDI'],y=ef['Forest Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[1,1].set_title('Forest Footprint')

axs[1,2].scatter(x=ef['HDI'],y=ef['Total Ecological Footprint'],c=ef['Region'],s=ef['Population (millions)'])



axs[1,2].set_title('Total Footprint')



for ax in axs.flat:

    ax.set(xlabel='HDI', ylabel='Total Ecological Footprint')



# Hide x labels and tick labels for top plots and y ticks for right plots.

for ax in axs.flat:

    ax.label_outer()

fig.suptitle('Footprint Components vs HDI by Country/Region')
# couldn't get the automatic legend to work using legend_elements() but Asia: green, North America: yellow