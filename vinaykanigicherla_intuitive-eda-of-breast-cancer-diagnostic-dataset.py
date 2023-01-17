"""
Importing all neccessary packages for data analysis and visualization.
"""
import numpy as np 
import pandas as pd
import seaborn as sns
import sklearn

from IPython.display import display
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
%matplotlib inline
"""
Loading the Dataset and Listing the Features/Attributes given in the dataset.
"""
complete_data = pd.read_csv('../input/data.csv')
complete_data = complete_data.drop('Unnamed: 32', axis=1) #drop Unnamed Column from dataset

display(complete_data.head()) #Show the first 5 rows of the dataset

print("Number of Data Points: {}".format(complete_data.shape[0])) #print number of data points
print("Number of Features/Attributes: {}".format(complete_data.shape[1])) # print number of features
print("Features/Attributes:", complete_data.columns) # print the list of all features in the dataset 
"""
Print description of the dataset. Includes Count, Mean, Standard Deviation, Minimum Value, 25th Percentile, 50th Percentile or Median, 
75th Percentile and Maximum Value for each of the Features in the Dataset
"""
complete_data.describe()
"""
Separate ID Data and Labels from Dataset
"""
id_data = complete_data['id'] #ID Column from complete_data
labels = complete_data['diagnosis'] #Lables column from complete_data

class_distribution = labels.value_counts() #distribution between malignant and benign tumors 
print (class_distribution)
"""
Form data by dropping unneccesary ID column from complete_data and dropping diagnosis column from complete_data as it serves as labels. 
Map labels/targets from letters to number
"""
data = complete_data.drop('id', axis=1) #drop the ID Column from complete_data
data = data.drop('diagnosis', axis=1) #drop the diagnosis column from complete_data
labels = labels.map({'B': 0, 'M': 1}) #Map the lables/targets vector, with 0 representing benign tumors and 1 represeting malignant tumors
"""
Plot Correlation Heatmap for data to observe the nature and extent correlation between various features in the dataset 
"""

plt.figure(figsize=(9,9)) 
sns.heatmap(data.iloc[:,:10].corr(),cbar=True,yticklabels=True,annot=True)
"""
Display first 5 columns of data to ensure that data does not contain unnecessary features and labels
"""
data.head()
"""
Scale the data and split it into training and testing sets
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler() #Create MinMaxScaler object from SKLearn
scaled_data = scaler.fit_transform(data) #Fit our data to the object to scale the data.

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.15) #Split data with 15% of data in the test set 
"""
Create dataset for finding contribution of individual features towards whether or not a certain cancer tumor is malignant or benign. 
"""
complete_data['diagnosis'] = complete_data['diagnosis'].map({'B': 0, 'M': 1}) #Map values in diagnosis column, 0 representing benign  and 1 represeting malignant 
data_for_corr = complete_data[['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'diagnosis']]
#Create data_for_corr with various features and diagnosis 
data_for_corr.head() 
"""
Create Pairplot from Seaborn to see relationship between individual features and diagnosis
"""
sns.pairplot(data_for_corr,palette='coolwarm',hue= 'diagnosis')
"""
Build Random Forest Classifier which we will use to find feature importances for the various features in the dataset.
"""
from sklearn.ensemble import RandomForestClassifier #import the classfier from SKLearn
rfc = RandomForestClassifier() #build the classfier
rfc.fit(X_train, y_train) #fit the model with our data
names = data.columns
# Print the results
print("Features sorted by their score:")
#print the features in descending order of feature importance
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True)) 
"""
Create plot showing variable feature importances of all features
"""
importance = rfc.feature_importances_ #isolate feature importances
sorted_importances = np.argsort(importance) #sort the feature importances
padding = np.arange(len(names)-1) + 0.5 #insert padding
plt.barh(range(len(sorted_importances)), importance[sorted_importances], align='center') #plot the data

#customize and show the plot
plt.yticks(padding, names[sorted_importances])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")
plt.show()
"""
Another and more complicated way of plotting feature importances using box plots and feature importances of features obtained from various 
different Random Forest models each with varying hyperparameters
"""
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

f1_scorer = make_scorer(f1_score, pos_label=0)
parameters = [{'n_estimators': np.arange(10,150,10), 'max_features': np.arange(5,30,5), 
                 'min_samples_split': np.arange(2, 8, 2)}]
rfc_grid = GridSearchCV(rfc, parameters, scoring = f1_scorer)

rfc_grid = rfc_grid.fit(X_train, y_train)
rfc_tuned = rfc_grid.best_estimator_
rfc_tuned.fit(X_train, y_train)

std = np.std([tree.feature_importances_ for tree in rfc_tuned.estimators_], axis=0)
indices = np.argsort(importance)[::-1]
label_features_sorted=[]
header = list(data.columns.values)
for i in indices:
    label_features_sorted.append(header[i])
    
plt.title("Feature importances")
plt.bar(range(data.shape[1]), importance[indices], color = "r", yerr = std[indices], align="center")
plt.xticks(range(data.shape[1]), label_features_sorted, rotation=90)
plt.xlim([-1, data.shape[1]])
plt.show
"""
Perform PCA (Principal Component Analysis) in order reduce the dimensionality of the data to print out the dimensionally reduced data (2D Data)
and use TSNE to visualize the same
"""
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

pca = PCA(n_components=2) #build the model
reduced_data = pca.fit_transform(data) #reduce the data, output is ndarray
print("Shape of Reduced Data: ", reduced_data.shape) #inspect shape of the `reduced_data`
print("Reduced Data:")
print(reduced_data)#print out the reduced data

#Plot the data reduced to 2 dimensions using TSNE with PCA
tsne = TSNE(init='pca')
tsne_plot = tsne.fit_transform(scaled_data)
plt.scatter(tsne_plot[:, 0], tsne_plot[:,1], c=labels)
plt.show
