# Basic Libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt # we only need pyplot

sb.set() # set the default Seaborn style for graphics



%matplotlib inline

import itertools



from PIL import Image

import PIL



import warnings

import plotly.offline as py

py.init_notebook_mode() # show output even if offline

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





cancerdata = pd.read_csv("/kaggle/input/data.csv")

cancerdata.head()
cancerdata.info()
cancerdata.drop('description', axis = 1, inplace = True)



cancerdata.head()
cancerdata.describe()
cancerdata.info()


diagnosis = pd.DataFrame(cancerdata['diagnosis'])

#to have 2 diagnosis format B&M and 0&1, diagnosis_bi stands for diagnosis_binary

diagnosis_bi = diagnosis.replace(to_replace = dict(M = 1, B = 0), inplace = False)

cancerdata['diagnosis_bi'] = diagnosis_bi



sb.countplot(cancerdata["diagnosis"])





trace = go.Pie(labels = ['benign','malignant'], values = cancerdata['diagnosis'].value_counts(), 

               textfont=dict(size=12), opacity = 1,

               marker=dict(colors=['green', 'red'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of diagnosis variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
# Import all mean, se, worst data and all

means = pd.DataFrame(cancerdata.loc[:,"radius_mean":"fractal_dimension_mean"])



se = pd.DataFrame(cancerdata.loc[:,"radius_se":"fractal_dimension_se"])



    

worst = pd.DataFrame(cancerdata.loc[:,"radius_worst":"fractal_dimension_worst"])



all_variables = pd.DataFrame(cancerdata.loc[:,"radius_mean":"fractal_dimension_worst"])

    

    # Draw the distributions of all 

f, axes = plt.subplots(30, 3, figsize=(18, 120))



count = 0

for var in all_variables:

    sb.boxplot(all_variables[var], orient = "h", ax = axes[count,0],color="r")

    sb.distplot(all_variables[var], ax = axes[count,1],color="g")

    sb.violinplot(all_variables[var], ax = axes[count,2])

    count += 1
meansDF = pd.concat([diagnosis, means], axis = 1 )



f, axes = plt.subplots(10, 1, figsize=(18, 40))



count = 0

for var in means:

    sb.swarmplot(x = var, y = "diagnosis", data = meansDF, orient = "h", ax = axes[count])

    count += 1
seDF = pd.concat([diagnosis, se], axis = 1 )#,join_axes= [y_train.index]



f, axes = plt.subplots(10, 1, figsize=(18 , 40))



count = 0

for var in se:

    sb.swarmplot(x = var, y = "diagnosis", data = seDF, orient = "h", ax = axes[count])

    count += 1
worstDF = pd.concat([diagnosis, worst], axis = 1 )#,join_axes= [y_train.index]



f, axes = plt.subplots(10, 1, figsize=(18 , 40))



count = 0

for var in worst:

    sb.swarmplot(x = var, y = "diagnosis", data = worstDF, orient = "h", ax = axes[count])

    count += 1




matrix = all_variables.corr()



matrix_col = matrix.columns.tolist()



matrix_array = np.array(matrix)



#print(variables_data.corr())



trace = go.Heatmap(z = matrix_array,

                   x = matrix_col,

                   y = matrix_col,

                   xgap = 2,

                   ygap = 2,

                   colorscale='agsunset',

                   colorbar   = dict() ,

                  )

layout = go.Layout(dict(title = 'Correlation Matrix for variables',

                        autosize = False,

                        height  = 870,

                        width   = 950,

                        margin  = dict(r = 0 ,l = 35,

                                       t = 70,b = 210,

                                     ),

                        yaxis   = dict(tickfont = dict(size = 14)),

                        xaxis   = dict(tickfont = dict(size = 14)),

                       )

                  )

fig = go.Figure(data = [trace],layout = layout)

py.iplot(fig)
# Heatmap of the Correlation Matrix

sb.set(font_scale=5)  #font size of coloumns and rows

f, axes = plt.subplots(1, 1, figsize=(80, 80))

all_heatmap = sb.heatmap(all_variables.corr(), vmin = -1, vmax = 1, linewidths = 1,

           annot = True, fmt = ".2f", annot_kws = {"size": 50}, cmap = "RdYlBu_r")



figure = all_heatmap.get_figure()

#figure.savefig('all_heatmap.png',dpi=25) #save heatmap as png
import matplotlib.patches as mpatches



#Set legend discription

B1=mpatches.Patch(color='green',label='Benign')

M1=mpatches.Patch(color='firebrick',label='Malignant')





correlated_1 = pd.DataFrame(cancerdata[["diagnosis_bi", "radius_mean", "perimeter_worst"]])

correlated_2 = pd.DataFrame(cancerdata[["diagnosis_bi", "perimeter_mean", "area_worst"]])

correlated_3 = pd.DataFrame(cancerdata[["diagnosis_bi", "radius_worst", "area_mean"]])

correlated_4 = pd.DataFrame(cancerdata[["diagnosis_bi", "area_mean", "area_worst"]])

sb.set(font_scale=1.5)  #font size of coloumns and rows

f, axes = plt.subplots(2, 2, figsize=(24 ,24))



axes[0, 0].scatter(x = "radius_mean", y = "perimeter_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi', data = correlated_1)

axes[0, 0].set_xlabel('radius_mean')

axes[0, 0].set_ylabel('perimeter_worst')

axes[0, 1].scatter(x = "perimeter_mean", y = "area_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = correlated_2)

axes[0, 1].set_xlabel('perimeter_mean')

axes[0, 1].set_ylabel('area_worst')

axes[1, 0].scatter(x = "radius_worst", y = "area_mean", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = correlated_3)

axes[1, 0].set_xlabel('radius_worst')

axes[1, 0].set_ylabel('area_mean')

axes[1, 1].scatter(x = "area_mean", y = "area_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = correlated_4)

axes[1, 1].set_xlabel('area_mean')

axes[1, 1].set_ylabel('area_worst')



axes[0, 0].legend(handles=[B1,M1])

axes[0, 1].legend(handles=[B1,M1])

axes[1, 0].legend(handles=[B1,M1])

axes[1, 1].legend(handles=[B1,M1])
uncorrelated_1 = pd.DataFrame(cancerdata[["diagnosis_bi", "fractal_dimension_worst", "area_mean"]])

uncorrelated_2 = pd.DataFrame(cancerdata[["diagnosis_bi", "fractal_dimension_worst", "radius_se"]])

uncorrelated_3 = pd.DataFrame(cancerdata[["diagnosis_bi", "texture_mean", "smoothness_worst"]])

uncorrelated_4 = pd.DataFrame(cancerdata[["diagnosis_bi", "texture_mean", "symmetry_se"]])



f, axes = plt.subplots(2, 2, figsize=(24 ,24))

sb.set(font_scale=1.5)  #font size of coloumns and rows

axes[0, 0].scatter(x = "fractal_dimension_worst", y = "area_mean", cmap = 'RdYlGn_r', c = 'diagnosis_bi', data = uncorrelated_1)

axes[0, 0].set_xlabel('fractal_dimension_worst')

axes[0, 0].set_ylabel('area_mean')

axes[0, 1].scatter(x = "fractal_dimension_worst", y = "radius_se", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = uncorrelated_2)

axes[0, 1].set_xlabel('fractal_dimension_worst')

axes[0, 1].set_ylabel('radius_se')

axes[1, 0].scatter(x = "texture_mean", y = "smoothness_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = uncorrelated_3)

axes[1, 0].set_xlabel('texture_mean')

axes[1, 0].set_ylabel('smoothness_worst')

axes[1, 1].scatter(x = "texture_mean", y = "symmetry_se", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = uncorrelated_4)

axes[1, 1].set_xlabel('texture_mean')

axes[1, 1].set_ylabel('symmetry_se')



axes[0, 0].legend(handles=[B1,M1])

axes[0, 1].legend(handles=[B1,M1])

axes[1, 0].legend(handles=[B1,M1])

axes[1, 1].legend(handles=[B1,M1])
#ne_correlated = negative correlation

ne_correlated_1 = pd.DataFrame(cancerdata[["diagnosis_bi", "fractal_dimension_mean", "perimeter_worst"]])

ne_correlated_2 = pd.DataFrame(cancerdata[["diagnosis_bi", "radius_worst", "smoothness_se"]])

ne_correlated_3 = pd.DataFrame(cancerdata[["diagnosis_bi", "fractal_dimension_mean", "radius_mean"]])

ne_correlated_4 = pd.DataFrame(cancerdata[["diagnosis_bi", "fractal_dimension_mean", "area_worst"]])



f, axes = plt.subplots(2, 2, figsize=(24 ,24))

sb.set(font_scale=1.5)  #font size of coloumns and rows

axes[0, 0].scatter(x = "fractal_dimension_mean", y = "perimeter_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi', data = ne_correlated_1)

axes[0, 0].set_xlabel('fractal_dimension_mean')

axes[0, 0].set_ylabel('perimeter_worst')

axes[0, 1].scatter(x = "radius_worst", y = "smoothness_se", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = ne_correlated_2)

axes[0, 1].set_xlabel('radius_worst')

axes[0, 1].set_ylabel('smoothness_se')

axes[1, 0].scatter(x = "fractal_dimension_mean", y = "radius_mean", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = ne_correlated_3)

axes[1, 0].set_xlabel('fractal_dimension_mean')

axes[1, 0].set_ylabel('radius_mean')

axes[1, 1].scatter(x = "fractal_dimension_mean", y = "area_worst", cmap = 'RdYlGn_r', c = 'diagnosis_bi' , data = ne_correlated_4)

axes[1, 1].set_xlabel('fractal_dimension_mean')

axes[1, 1].set_ylabel('area_worst')



axes[0, 0].legend(handles=[B1,M1])

axes[0, 1].legend(handles=[B1,M1])

axes[1, 0].legend(handles=[B1,M1])

axes[1, 1].legend(handles=[B1,M1])
# Import KMeans from sklearn.cluster

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score





correlated = pd.DataFrame(cancerdata[["radius_mean", "perimeter_worst", "perimeter_mean", "area_worst", "radius_worst", 

                                      "area_se", "perimeter_se", "radius_se", "concavity_mean", "concave_points_mean", "area_mean"]])





# Vary the Number of Clusters

min_clust = 1

max_clust = 40

init_algo = 'k-means++'





# Compute Within Cluster Sum of Squares

within_ss = []

for num_clust in range(min_clust, max_clust+1):

    kmeans = KMeans(n_clusters = num_clust, init = init_algo, n_init = 5)

    kmeans.fit(correlated)

    within_ss.append(kmeans.inertia_)

    

sb.set(font_scale=1)  #font size of coloumns and rows



# Angle Plot : Within SS vs Number of Clusters

f, axes = plt.subplots(1, 1, figsize=(16,4))

plt.plot(range(min_clust, max_clust+1), within_ss)

plt.xlabel('Number of Clusters')

plt.ylabel('Within Cluster Sum of Squares')

plt.xticks(np.arange(min_clust, max_clust+1, 1.0))

plt.grid(which='major', axis='y')

plt.show()




# Set "optimal" Clustering Parameters

num_clust = 2

init_algo = 'k-means++'



# Create Clustering Model using KMeans

kmeans = KMeans(n_clusters = num_clust,         

               init = init_algo,

               n_init = 20)                 



# Fit the Clustering Model on the Data

kmeans.fit(correlated)
# Print the Cluster Centers

print("Features", "\tradius_mean", "\tperimeter_worst", "\tperimeter_mean", "\tarea_worst", "\tradius_worst", 

      "\tarea_se", "\tperimeter_se", "\tradius_se", "\tconcavity_mean", "\tconcave_points_mean", "\tarea_mean")



for i, center in enumerate(kmeans.cluster_centers_):

    print("Cluster", i, end=":\t")

    for coord in center:

        print(round(coord, 2), end="\t\t")

    print()

print()



# Print the Within Cluster Sum of Squares

print("Within Cluster Sum of Squares :", kmeans.inertia_)

print()



# Predict the Cluster Labels

labels = kmeans.predict(correlated)



# Append Labels to the Data

correlated_labeled = correlated.copy()

correlated_labeled["Cluster"] = pd.Categorical(labels)



sb.set(font_scale=1) 

f, axes = plt.subplots(1, 2, figsize=(12, 4))

# Summary of the Cluster Labels

sb.countplot(correlated_labeled["Cluster"],ax = axes[0])

#heatmap for Kmean

from sklearn.metrics import confusion_matrix



sb.heatmap(confusion_matrix(cancerdata['diagnosis_bi'].values.tolist(), labels.tolist()),

           annot = True, fmt = 'd', annot_kws = {'size': 18},ax = axes[1])





#accuracy score

kmean_accuracy = accuracy_score(cancerdata['diagnosis_bi'].values.tolist(), labels.tolist())

print("Accuracy of cluster =",kmean_accuracy)

# Plot the Clusters on 2D grids

sb.pairplot(correlated_labeled, vars = correlated.columns.values, hue="Cluster")



#plt.savefig('correlated_pairplot.png',dpi=100) #save image
from sklearn.metrics import plot_roc_curve

from sklearn.metrics import plot_precision_recall_curve

from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split



# split data train 70 % and test 30 %

x_train, x_test, y_train, y_test = train_test_split(correlated, diagnosis, test_size=0.3, random_state = 100)



def show_metrics():

    tp = cm_test[1,1]

    fn = cm_test[1,0]

    fp = cm_test[0,1]

    tn = cm_test[0,0]

    print('For test prediction')

    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))

    print('Precision =     {:.3f}'.format(tp/(tp+fp)))

    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))

    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/

                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))
# Import essential models and functions from sklearn

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz

import graphviz

from graphviz import Graph

# Decision Tree using Train Data

dectree = DecisionTreeClassifier(max_depth = 7, random_state = 100)  # create the decision tree object

dectree.fit(x_train, y_train)                    # train the decision tree model



# Predict Response corresponding to Predictors

y_train_pred = dectree.predict(x_train)

y_test_pred = dectree.predict(x_test)



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", dectree.score(x_train, y_train))

print()

dectree_acc=dectree.score(x_test, y_test)

# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Classification Accuracy \t:", dectree.score(x_test, y_test))

print()



# Plot the Confusion Matrix for Train and Test



cm_train = confusion_matrix(y_train, y_train_pred)

cm_test = confusion_matrix(y_test, y_test_pred)





# Plot the Confusion Matrix for Train and Test

sb.set(font_scale=1) 

f, axes = plt.subplots(1, 2, figsize=(12, 4))

sb.heatmap(cm_train,

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])

sb.heatmap(cm_test, 

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])





# Plot the Decision Tree

treedot = export_graphviz(dectree,                                      # the model

                          feature_names = x_train.columns,              # the features 

                          out_file = None,                              # output file

                          filled = True,                                # node colors

                          rounded = True,                               # make pretty

                          special_characters = True)                    # postscript



graphviz.Source(treedot)

#graph.format = 'png'

#graph.render('dtree_render',view=True)#save image


show_metrics()



plot_roc_curve(dectree,x_test, y_test)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix



from sklearn.model_selection import KFold



#random forest classifier with n_estimators=100 (default)

clf_rf = RandomForestClassifier(n_estimators= 40, max_depth = 7, oob_score = True, random_state = 100)      



clr_rf = clf_rf.fit(x_train,y_train.values.ravel())



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", clf_rf.score(x_train, y_train))

print()

rf_acc = clf_rf.score(x_test, y_test)

# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Classification Accuracy \t:", clf_rf.score(x_test, y_test))

print()





print('Out of bag Score is: ', clf_rf.oob_score_) # mean prediction error



y_train_pred = clf_rf.predict(x_train)

y_test_pred = clf_rf.predict(x_test)



cm_train = confusion_matrix(y_train, y_train_pred)

cm_test = confusion_matrix(y_test, y_test_pred)





# Plot the Confusion Matrix for Train and Test

sb.set(font_scale=1) 

f, axes = plt.subplots(1, 2, figsize=(12, 4))

sb.heatmap(cm_train,

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])

sb.heatmap(cm_test, 

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])





show_metrics()



rf_roc = plot_roc_curve(clr_rf,x_test, y_test)

from sklearn.feature_selection import RFECV

import plotly.express as px



x2_train, x2_test, y2_train, y2_test = train_test_split(all_variables, diagnosis, test_size=0.3, random_state = 100)



n_features = RandomForestClassifier(n_estimators= 40, max_depth = 7, random_state = 100) 

rfecv = RFECV(estimator=n_features, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(x2_train, y2_train.values.ravel())

rfecv.ranking_[8]

rfecv_para =x2_train.columns[rfecv.support_]



print('Ideal number of features :', rfecv.n_features_)

print('Best features :',rfecv_para )



#sb.set(font_scale=1) 

#plt.figure(figsize=(16, 9))

#plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)

#plt.xlabel('Number of features selected', fontsize=14, labelpad=20)

#plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)

#plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='red', linewidth=3)



#show the graph using plotly

fig = px.line( x= range(1, len(rfecv.grid_scores_) + 1),y= rfecv.grid_scores_,labels={'x':'n_features', 'y':'scoring'})



fig.update_layout(title='Recursive Feature Elimination with Cross-Validation',

                   xaxis_title='Number of features selected',

                   yaxis_title='% Correct Classification',

                 )





fig.show()
RFECV_features = pd.DataFrame(cancerdata[rfecv_para])



# split data train 70 % and test 30 %

x2_train, x2_test, y2_train, y2_test = train_test_split(RFECV_features, diagnosis, test_size=0.3, random_state = 100)



#random forest classifier with n_estimators=10 (default)

clf_rf_2 = RandomForestClassifier(n_estimators= 40, max_depth = 7, oob_score = True, random_state = 100)      



clr_rf_2 = clf_rf_2.fit(x2_train,y2_train.values.ravel())





# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", clr_rf_2.score(x2_train, y2_train))

print()

rfecv_acc = clr_rf_2.score(x2_test, y2_test)

# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Classification Accuracy \t:", clr_rf_2.score(x2_test, y2_test))

print()





print('Out of bag Score is: ', clr_rf_2.oob_score_) # mean predicition error



y2_train_pred = clr_rf_2.predict(x2_train)

y2_test_pred = clr_rf_2.predict(x2_test)



cm_train = confusion_matrix(y2_train, y2_train_pred)

cm_test = confusion_matrix(y2_test, y2_test_pred)





# Plot the Confusion Matrix for Train and Test

sb.set(font_scale=1) 

f, axes = plt.subplots(1, 2, figsize=(12, 4))

sb.heatmap(cm_train,

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])

sb.heatmap(cm_test, 

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])
show_metrics()



rfecv_roc = plot_roc_curve(clr_rf_2,x2_test, y2_test,name='RFECV')

ax=plt.gca()

plot_roc_curve(clr_rf,x_test, y_test,ax=ax)

rfecv_roc.plot(ax=ax,name = 'RFECV')
cancer_pred = cancerdata.sample(n=5) #select 5random data

cancer_pred





correlated_pred = pd.DataFrame(cancer_pred[rfecv_para])

diagnosis_pred = clr_rf_2.predict(correlated_pred)

# Summarize the Actuals and Predictions

diagnosis_pred = pd.DataFrame(diagnosis_pred, columns = ["PredDiagnosis"], index = cancer_pred.index)

cancer_correlated_acc = pd.concat([cancer_pred[["id", "diagnosis"]], diagnosis_pred], axis = 1)







# Predict Probabilities corresponding to Predictors

diagnosis_prob = clf_rf_2.predict_proba(correlated_pred)







diagnosis_prob = pd.DataFrame(list(diagnosis_prob[:,1]), columns = ["ProDiagnosis"], index = cancer_pred.index)

cancer_correlated_conf = pd.concat([cancer_correlated_acc, diagnosis_prob], axis = 1)



cancer_correlated_conf
from sklearn.metrics import confusion_matrix

from sklearn import svm

from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score









svc = svm.SVC()  

svc.fit(x_train, y_train.values.ravel())                    



                   



# Predict Response corresponding to Predictors

y_train_pred = svc.predict(x_train)

y_test_pred = svc.predict(x_test)



# Check the Goodness of Fit (on Train Data)

print("Goodness of Fit of Model \tTrain Dataset")

print("Classification Accuracy \t:", svc.score(x_train, y_train))

print()

svc_acc = svc.score(x_test, y_test)

# Check the Goodness of Fit (on Test Data)

print("Goodness of Fit of Model \tTest Dataset")

print("Classification Accuracy \t:", svc.score(x_test, y_test))

print()



cm_train = confusion_matrix(y_train, y_train_pred)

cm_test = confusion_matrix(y_test, y_test_pred)





# Plot the Confusion Matrix for Train and Test

sb.set(font_scale=1) 

f, axes = plt.subplots(1, 2, figsize=(12, 4))

sb.heatmap(cm_train,

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[0])

sb.heatmap(cm_test, 

           annot = True, fmt=".0f", annot_kws={"size": 18}, ax = axes[1])



show_metrics()



plot_roc_curve(svc,x_test, y_test)
x = ["Kmean accuracy","Dectree accuracy","RandomForest accuracy","SVC accuracy","RFECV accuracy"]

y = [kmean_accuracy,dectree_acc,rf_acc,svc_acc,rfecv_acc]



fig = go.Figure()

#fig = px.bar(x=x,y=y)

fig.add_trace(go.Histogram(histfunc="sum", y=y, x=x, name="sum"))

#fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()