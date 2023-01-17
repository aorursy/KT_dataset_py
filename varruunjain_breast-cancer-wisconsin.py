# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA

import seaborn as sns

from sklearn.feature_selection import SelectFromModel

from subprocess import check_output

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2, f_classif

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score#print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
data=pd.read_csv("../input/data.csv")

data.head(n=1)

total_records=data.shape[0]

mal_records=(data[data.diagnosis!='M']).shape[0]

ben_records=(data[data.diagnosis!='B']).shape[0]

print("Total Malignant recorded {}".format(mal_records))

print("Total Benign Patients recorded {}".format(ben_records))

print("Total Patients records {}".format(total_records))

#pd.unique(data.diagnosis)

#sns.countplot(data['diagnosis'],label="Count")



#features and labels

label_raw=data.diagnosis

features_raw=data.drop(["diagnosis","id","Unnamed: 32"],axis=1)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler

scale=MinMaxScaler()

#scale=RobustScaler()

number=features_raw.columns

features=pd.DataFrame(index=features_raw.index, columns=number)

features[number]=scale.fit_transform(features_raw[number])

le=LabelEncoder()

label=list(le.fit_transform(label_raw))





sns.boxplot(data=features, order=None, hue_order=None, orient='h', color=None, palette=None, saturation=0.85, width=0.8, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None)

#SVM , KNN, Decision tree, Random forest

def train(X,y):



    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.tree import DecisionTreeClassifier

    from sklearn.ensemble import AdaBoostClassifier ,RandomForestClassifier

    from sklearn.svm import SVC

    from sklearn.metrics import accuracy_score, fbeta_score, f1_score, log_loss

    from sklearn.model_selection import train_test_split



    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=None)

    clf_1=KNeighborsClassifier()

    clf_2=DecisionTreeClassifier()

    clf_3=SVC()

    clf_4=RandomForestClassifier()

    clf=[]

    accuracy=[]

    fscore=[]

    loss=[]

    for classifier  in [clf_1,clf_2,clf_3,clf_4]:

        clf.append(classifier.__class__.__name__)

        classifier.fit(X_train,y_train)

        predict=classifier.predict(X_test)

        accuracy_test=accuracy_score(y_test,predict)

        accuracy.append(accuracy_test)

        #f_score=fbeta_score(y_test,predict,beta=0.5)

        f_score=f1_score(y_test,predict)

        loss_func=log_loss(y_test,predict)

    

        fscore.append(f_score)

        loss.append(loss_func)

    '''import graphviz 

    from sklearn import tree

    clf_2.fit(X_train,y_train)

    dot_data = tree.export_graphviz(clf_2, out_file=None,feature_names=features.columns,class_names=data.diagnosis) 

    graph = graphviz.Source(dot_data) 

    graph.render("data") 

    print(graph)

    '''

    

    

    #print(loss)

    print("{} F_score {:.3f} and Log loss {:.3f}".format(clf[0],fscore[0],loss[0]))

    print("Accuracy {:.3f}".format(accuracy[0]))

    print("{} F_score {:.3f} and Log loss {:.3f}".format(clf[1],fscore[1],loss[1]))

    print("Accuracy {:.3f}".format(accuracy[1]))

    print("{} F_score {:.3f} and Log loss {:.3f}".format(clf[2],fscore[2],loss[2]))

    print("Accuracy {:.3f}".format(accuracy[2]))

    print("{} F_score {:.3f} and Log loss {:.3f}".format(clf[3],fscore[3],loss[3]))

    print("Accuracy {:.3f}".format(accuracy[3]))





    
# XG BOOST feature importances

def train_optimise(clf,X,y):

    

    from sklearn.metrics import accuracy_score, fbeta_score, f1_score, log_loss

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test=train_test_split(features,label,test_size=0.2,random_state=None)

    from xgboost import XGBClassifier

    from xgboost import plot_importance

    model=clf

    model.fit(X_train, y_train)

    predict_train=model.predict(X_train)

    predict_test=model.predict(X_test)

    

    print("{} accuracy score {:.4f}".format(model.__class__.__name__,accuracy_score(predict_test,y_test)))

    print("{} f_beta score {:.4f}".format(model.__class__.__name__,fbeta_score(predict_test,y_test,beta=0.5)))

    print("{} f1 score {:.4f}".format(model.__class__.__name__,f1_score(predict_test,y_test)))

    

    # feature importance

    feature_imp=pd.DataFrame(model.feature_importances_,index=features.columns,columns=["Importance"])

    feature_imp_sort=(feature_imp.sort_values(['Importance']))

    print(feature_imp_sort)

    plot_importance(model)

    

    



#PCA feature decomposition

def pcA():

    print("======PCA=====")

    pca=PCA(n_components=25,  copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

    feature_transform_pca=pca.fit_transform(features)

    var=pca.explained_variance_ratio_

    plt.plot(var)

    plt.title("Variance trade off curve")

    plt.show()

    plt.title("Cummulative Variance Curve")

    var1=np.cumsum(pca.explained_variance_ratio_)

    plt.plot(var1)

    plt.show()

    print(feature_transform_pca.shape)

    return feature_transform_pca

    #feature_transform_pca.shape



#feature Selection

def feature_sel():  

    print("=====SelectKBest=======(Univariate)")

    from sklearn.feature_selection import SelectKBest

    from sklearn.feature_selection import chi2, f_classif

    feat_new=features

    X_new=SelectKBest(f_classif,k=10).fit_transform(feat_new,label)

    features_X=pd.DataFrame(X_new)

    

    print(features_X.shape)

    return features_X

def reduction_feature():

    print("=====Recursive feature Elimination=====")

    from sklearn.feature_selection import RFE

    from sklearn.tree import DecisionTreeClassifier

    clf_r=DecisionTreeClassifier()

    feat=features

    c=RFE(clf_r).fit_transform(feat,label)

    

    print(c.shape)

    

    return c
#Results without any feature reduction or dimensionality reduction

train(features,label)
# Results feature reudction using SelectKbest

FR=feature_sel()

train(FR,label)

#Results feature reduction using RFE

rf=reduction_feature()

train(rf,label)
#Results after dimensionality reduction (Principal Component Analysis)

Feature_pca=pcA()

train(Feature_pca,label)
#Results of XGBoost without feature reduction

train_optimise(XGBClassifier(),features,label)
#results using Logistic Regression



X_train,X_test,y_train,y_test=train_test_split(Feature_pca,label,test_size=0.2,random_state=None)

anova_filter = SelectKBest(f_classif, k=5)

clf = LogisticRegression()

anova_svm = Pipeline([('anova', anova_filter), ('lr', clf)])

# You can set the parameters using the names issued

# For instance, fit using a k of 10 in the SelectKBest

# and a parameter 'C' of the svm

anova_svm.set_params(anova__k=10, lr__C=0.1).fit(X_train, y_train)

prediction = list(anova_svm.predict(X_test))

accuracy_score(prediction,y_test)
