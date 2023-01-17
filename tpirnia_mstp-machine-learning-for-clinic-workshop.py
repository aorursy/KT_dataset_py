#General packages.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #plotting

import seaborn as sns #more plotting



import os #for interacting with the operating system. 



from subprocess import call

from IPython.display import Image



import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility



pd.options.mode.chained_assignment = None  #hide any pandas warnings



# Any results you write to the current directory are saved as output.
#ML related packages:

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix as cm

from sklearn.metrics import accuracy_score as score



from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz #plot tree

from sklearn.metrics import roc_curve, auc #for model evaluation

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance
#setting the data path and file name:

data_path = "../input/logistic-regression-heart-disease-prediction" 

fname = os.listdir(data_path)[0] #file name



#loading and examination of data set.

dataset = pd.read_csv((data_path +'/'+ fname))

dataset.rename(columns={'male':'sex'}, inplace=True)



#remove entries with NANs

dataset.dropna(axis=0,inplace=True)



dataset.head() #show first few entries.
#Separate data set into X (input features) and y (output label).

X = dataset.drop('TenYearCHD', 1) #dataset.iloc[:,:-1].values

y = dataset['TenYearCHD'] #dataset.iloc[:,-1].values



features = dataset.drop('TenYearCHD', 1).columns

# print(features.tolist())
def plot_hists(dataframe, features, rows, cols):

    ''' for plotting histogram of each feature'''

    fig=plt.figure(figsize=(25,25))

    

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title('Distribution: '+ feature,color='DarkRed')

        

    fig.tight_layout()  

    plt.show()



plot_hists(dataset,dataset.columns,5,4)

#Split data set into train and test sets. Our models will be trained 

# using the former and evaluated on the test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify=y)

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)
#logistic regression to predict ten year risk of CHD.

LR_clf = LogisticRegression(solver='liblinear')#random_state=0, solver='lbfgs')

LR_clf.fit(X_train, y_train)

y_LR_pred = LR_clf.predict(X_test)

y_LR_pred_quant = LR_clf.predict_proba(X_test)[:, 1]
#logistic regression result analysis::

cm_LR=cm(y_test,y_LR_pred)

# print('Confusion matrix: ',cm_LR)

sns.heatmap(cm_LR,annot=True)

plt.show()



fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test, y_LR_pred_quant)

fig, ax = plt.subplots()

ax.plot(fpr_LR, tpr_LR)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for CHD: Logistic Regression classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

plt.show()

print('Accuracy:',score(y_test,y_LR_pred)*100)

sensitivity = cm_LR[0,0]/(cm_LR[0,0]+cm_LR[1,0])

print('Sensitivity : ', sensitivity )

specificity = cm_LR[1,1]/(cm_LR[1,1]+cm_LR[0,1])

print('Specificity : ', specificity)

print('AUC:',auc(fpr_LR, tpr_LR))
#logistic regression to predict ten year risk of CHD.

LR_clf = LogisticRegression(solver='liblinear', class_weight='balanced')#random_state=0, solver='lbfgs')

LR_clf.fit(X_train, y_train)

y_LR_pred = LR_clf.predict(X_test)

y_LR_pred_quant = LR_clf.predict_proba(X_test)[:, 1]
# feature_names = [i for i in X.columns]

y_train_str = y_train.astype('str')

y_train_str[y_train == 0] = 'no disease'

y_train_str[y_train == 1] = 'CHD'

y_train_str = y_train_str.values
#random forest model of ten year risk of CHD.

model_RF = RandomForestClassifier(max_depth=4, class_weight='balanced', criterion='entropy')#, n_estimators=4)

model_RF.fit(X_train, y_train)

estimator = model_RF.estimators_[1]
# pd.get_dummies(dataset, drop_first=True)

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = features,

                class_names = ['no disease', 'CHD'],

                rounded = True, proportion = True, 

                label='all',

                precision = 7, filled = True, impurity=False)



call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

Image(filename = 'tree.png')

y_RF_predict = model_RF.predict(X_test)

y_RF_pred_quant = model_RF.predict_proba(X_test)[:, 1]

y_RF_pred_bin = model_RF.predict(X_test)

print('Accuracy:',score(y_test,y_RF_predict)*100)

confusion_matrix_ = cm(y_test, y_RF_pred_bin)

confusion_matrix_

total=sum(sum(confusion_matrix_))

print(confusion_matrix_)

sensitivity = confusion_matrix_[0,0]/(confusion_matrix_[0,0]+confusion_matrix_[1,0])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix_[1,1]/(confusion_matrix_[1,1]+confusion_matrix_[0,1])

print('Specificity : ', specificity)

fpr_RF, tpr_RF, thresholds = roc_curve(y_test, y_RF_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr_RF, tpr_RF)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for CHD: Random Forest classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)

print('AUC:',auc(fpr_RF, tpr_RF))
fig, ax = plt.subplots(figsize=(10,10))

ax.plot(fpr_LR, tpr_LR, color='green', label='Logistic Reg')

ax.plot(fpr_RF, tpr_RF, color='blue', label='Random Forest')

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for CHD: Model Comparison')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.legend()

plt.grid(True)

plt.show()
perm = PermutationImportance(model_RF, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X.columns.tolist())
shap.initjs()

explainer = shap.TreeExplainer(model_RF)

shap_values = explainer.shap_values(X_test)



# shap.summary_plot(shap_values[1], X_test, plot_type="bar")

shap_values = explainer.shap_values(X_train[:50])

shap.force_plot(explainer.expected_value[1], shap_values[1], X_test[:50])