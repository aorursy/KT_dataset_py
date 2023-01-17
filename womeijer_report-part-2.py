import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

from scipy import stats, linalg

import seaborn as sns

mydata = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])

#make a table of the data!

#categorize/clear the data

#zip codes are strings
mydata['zipcode'] = mydata['zipcode'].astype(str)
#the other non ratio data are "categories" thanks pandas :D
mydata['waterfront'] = mydata['waterfront'].astype('category',ordered=True)
mydata['condition'] = mydata['condition'].astype('category',ordered=True)
mydata['view'] = mydata['view'].astype('category',ordered=True)
mydata['grade'] = mydata['grade'].astype('category',ordered=False)

#drop ID
mydata = mydata.drop(['id', 'date'],axis=1)
mydata = mydata.dropna()
mydata = mydata[mydata.bedrooms < 15]
#display a table of all the data for refernce (handy)
df = pd.DataFrame(data = mydata)
str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in mydata.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)
# Get to the numeric columns by inversion            
num_list = mydata.columns.difference(str_list) 
# Create Dataframe containing only numerical features
numData = mydata[num_list]

#and then remove more stuff
interestingCol =['price','bedrooms','bathrooms','sqft_above','sqft_living']
numData = numData[interestingCol]
originalData = numData.copy()

#reduce the number of data points
numData = numData.sample(n=11000, random_state = 13)
#figure out what the standardized million dollars is and save that
oneMillSTD = (numData['price'].median()-numData['price'].mean())/numData['price'].std()
numData =(numData - numData.mean()) / numData.std()




numData.fillna(method='backfill', inplace=True)

numData.describe()
originalData.head()


X = numData.drop(['price'],axis=1)
y = numData['price']

attributeNames = list(X)

classNames = ['MillionDollarHome','notMDH']

from sklearn import model_selection
X, X_testglb, y, y_testglb = model_selection.train_test_split(X,y, test_size = (1/11),random_state = 42)

N, M = X.shape
Xpd = X.copy()
X = X.as_matrix()

from sklearn import tree

#first we have to create a new y for million dollars or not!
def millionDollars(money):
    #returns false if the price is less than a million
    #returns true if the price is equal to or greater than a million dollars
    if(money < oneMillSTD):
        return 0
    else:
        return 1

#create the new classification data set
y_dtc = y.apply(millionDollars)

y_dtc = y_dtc.as_matrix()
yCopy = y.copy()
y = y_dtc

#use the SKLEARN decision tree maker
dtc = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=1000)
dtc = dtc.fit(X,y_dtc)

#visualize
#code borroed from 
#https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset
from IPython.display import Image as PImage
from subprocess import check_call

def drawTree(datree):
    with open("tree1.dot", 'w') as f:
         f = tree.export_graphviz(datree,
                              out_file=f,
                              rounded = True,
                              filled= True )
    #Convert .dot to .png to allow display in web notebook
    check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])
    return("tree1.png")
    
#use the function to draw the tree
PImage(drawTree(dtc))

###some code qualifying the data


### some optimization code using cross validation!?
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 41, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = sum(np.abs(y_est_test - y_test)) / float(len(y_est_test))
        misclass_rate_train = sum(np.abs(y_est_train - y_train)) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

    

plt.boxplot(Error_test.T)
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Test error across CV folds, K={0})'.format(K))
plt.show()


plt.plot(tc, Error_train.mean(1))
plt.plot(tc, Error_test.mean(1))
Error_tot = Error_train.mean(1) + Error_test.mean(1)
plt.plot(tc, Error_tot)
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate, CV K={0})'.format(K))
plt.legend(['Error_train','Error_test','Error_total'])
optimalDepth = np.argmin(Error_tot)
print(optimalDepth)

CV = model_selection.KFold(n_splits=2,shuffle=True)

for train_index, test_index in CV.split(X):
    #print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=optimalDepth)
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)


#y_est_test = dtc.predict(X_testglb)

#using the confusion matrix not actually graphically
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

cm = confusion_matrix(y_test,y_est_test)
print('Confusion matrix: \n',cm)
sns.heatmap(cm, annot=True)
plt.title('Optimal Decision Tree confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

percision = (precision_score(y_test,y_est_test))
recall =(recall_score(y_test,y_est_test))
print( "%.2f" % percision)
print( "%.2f" % recall)
#PImage(drawTree(dtc))

y_optdtc = y_est_test

from sklearn import neighbors
from sklearn import model_selection

#max number of neighbors
L = 40

CV = model_selection.KFold(n_splits=40)
errors = np.zeros((N,L))
i=0

for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,L))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y_dtc[train_index]
    X_test = X[test_index,:]
    y_test = y_dtc[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = neighbors.KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i,l-1] = np.sum(y_est[0]!=y_test[0])

    i+=1

errorOfKNN = 100*sum(errors,0)/N
plt.plot(errorOfKNN)
plt.xlabel('Number of neighbors')
plt.ylabel('Classification error rate (%)')

optimalK = np.argmin(errorOfKNN)
print(optimalK)
print(min(errorOfKNN*100))

knclassifier = neighbors.KNeighborsClassifier(n_neighbors=5);
knclassifier.fit(X, y);

CV = model_selection.KFold(n_splits=2)
i = 0
for train_index, test_index in CV.split(X):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,L))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y_dtc[train_index]
    X_test = X[test_index,:]
    y_test = y_dtc[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    
    knclassifier = neighbors.KNeighborsClassifier(n_neighbors=optimalK);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    cm = confusion_matrix(y_test,y_est)
    print('Confusion matrix: \n',cm)
    i += 1


y_optKnn = y_est
sns.heatmap(cm, annot=True)
plt.title('Optimal Nearest Neighbor confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
percision = (precision_score(y_test,y_est))
recall =(recall_score(y_test,y_est))
print( "%.2f" % percision)
print( "%.2f" % recall)
from sklearn.linear_model import LogisticRegression
  
    

from sklearn import datasets
from sklearn.feature_selection import RFE, RFECV

#redfine X to get a pandas dictionary


rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features: %d" % rfecv.n_features_)
#print('Selected features: %s' % list(attributeNames[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print(rfecv.support_)
optimalCol = ['bedrooms', 'sqft_above','sqft_living']
Xreg = Xpd[optimalCol].as_matrix()
logModel = LogisticRegression()

#max number of neighbors
K = 40

CV = model_selection.KFold(n_splits=2)
errors = np.zeros((N,L))
Error_logreg = np.empty((K,1))


k=0
for train_index, test_index in CV.split(Xreg):
    print('CV-fold {0} of {1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = Xreg[train_index,:]
    y_train = y[train_index]
    X_test = Xreg[test_index,:]
    y_test = y[test_index]
    model = LogisticRegression(C=N)
    model.fit(X_train,y_train)
    y_logreg = model.predict(X_test)
    Error_logreg[k] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
    cm = confusion_matrix(y_test,y_logreg)
    print('Confusion matrix: \n',cm)
    k+=1


y_globaltest = y_test
y_optlog = y_logreg
sns.heatmap(cm, annot=True)
plt.title('Optimal Nearest Neighbor confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

percision = (precision_score(y_test,y_logreg))
recall =(recall_score(y_test,y_logreg))
print( "%.2f" % percision)
print( "%.2f" % recall)
from matplotlib.colors import ListedColormap

plotCol = ['sqft_above', 'sqft_living']
Xplot = Xpd[plotCol].as_matrix()

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
h = .02  # step size in the mesh
model.fit(Xplot,y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = Xplot[:, 0].min() - .5, Xplot[:, 0].max() + .5
y_min, y_max = Xplot[:, 1].min() - .5, Xplot[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(10,6))
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(Xplot[:, 0], Xplot[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel('sqft_above')
plt.ylabel('sqft_living')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


plt.show()



clf = neighbors.KNeighborsClassifier(n_neighbors = 5)
clf.fit(Xplot, y)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = Xplot[:, 0].min() - 1, Xplot[:, 0].max() + 1
y_min, y_max = Xplot[:, 1].min() - 1, Xplot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(Xplot[:, 0], Xplot[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("sqft_living")
plt.ylabel("bedrooms")
plt.title("House Price Classifier")

plt.show()
import sklearn.linear_model as lm
K = 40
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
Error_logreg = np.empty((K,1))
Error_nearestn = np.empty((K,1))
n_tested=0



k=0
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]
    X_trainreg = Xreg[train_index,:]
    X_testreg = Xreg[test_index,:]

    # Fit and evaluate Logistic Regression classifier
    model = lm.logistic.LogisticRegression(C=N)
    model = model.fit(X_trainreg, y_train)
    y_logreg = model.predict(X_testreg)
    Error_logreg[k] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
    
    # Fit and evaluate Decision Tree classifier
    model2 = neighbors.KNeighborsClassifier(n_neighbors=optimalK);
    model2 = model2.fit(X_train, y_train)
    y_nearestn = model2.predict(X_test)
    Error_nearestn[k] = 100*(y_nearestn!=y_test).sum().astype(float)/len(y_test)

    k+=1

# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_nearestn)
# and test if the p-value is less than alpha=0.05. 
z = (Error_logreg-Error_nearestn)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
plt.figure()
plt.boxplot(np.concatenate((Error_logreg, Error_nearestn),axis=1))
plt.xlabel('Logistic Regression   vs.   Nearest Neighbor')
plt.ylabel('Cross-validation error [%]')

plt.show()

from scipy import stats, integrate
sns.distplot(Error_logreg, label="Logistic Regression", hist = False, rug = True)
sns.distplot(Error_nearestn, label ="Nearest Neighbor", hist = False, rug = True)


print(sum(y))
print(10000-sum(y))
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

fpr,tpr,_ = roc_curve(y_globaltest.ravel(),y_est_test.ravel())

#y_optlog = y_logreg

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: Decision Tree')
plt.legend(loc="lower right")
plt.show()
print(y_logreg)
