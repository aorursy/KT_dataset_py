import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# apply ignore

import warnings

warnings.filterwarnings('ignore')
#load train data

train_data = pd.read_csv('../input/learn-together/train.csv')

train_data.head()
# Select columns 

selected_features = [cname for cname in train_data.columns if cname not in ['Id','Cover_Type']]



X = train_data[selected_features]

y = train_data.Cover_Type
from sklearn.model_selection import train_test_split



y_labels = np.unique(train_data.Cover_Type)



# Break off validation set from training data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline

from sklearn.metrics import confusion_matrix

from pydoc import locate



def build_model(model):

    # Fit to data and predict using pipelined scaling

    classifier = make_pipeline(StandardScaler(), model)

    classifier.fit(X_train, y_train)

    return classifier



def make_report(model, y_pred):

    from sklearn.metrics import classification_report

    print(classification_report(y_val, y_pred))

    

def plot_confusion_matrix(model, y, y_pred, normalized=True, cmap='bone'):

    classes = np.sort(y.unique())

    # run trained model.

    cm = confusion_matrix(y, y_pred)

    #df = pd.DataFrame(cm, columns)

    labels = y_labels.astype('str')

    df = pd.DataFrame(cm, columns=labels, index=labels)

    print(df)

   

def print_report(className, params=None):

    print('\n   '+className+'\n')

    the_class = locate(className)

    model = the_class() if params==None else the_class(**params)

    #print(model)

    classifier = build_model(model)

    y_prd = classifier.predict(X_val)

    make_report(classifier, y_prd)

    plot_confusion_matrix(model, y_val, y_prd)    
# Inherently multiclass

print_report('sklearn.naive_bayes.BernoulliNB')

print_report('sklearn.tree.DecisionTreeClassifier')

print_report('sklearn.tree.ExtraTreeClassifier')

print_report('sklearn.naive_bayes.GaussianNB')

print_report('sklearn.neighbors.KNeighborsClassifier')

print_report('sklearn.semi_supervised.LabelPropagation')

print_report('sklearn.semi_supervised.LabelSpreading')

print_report('sklearn.discriminant_analysis.LinearDiscriminantAnalysis')

print_report('sklearn.svm.LinearSVC', {'multi_class':"crammer_singer"} )

# solver : ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ 

print_report('sklearn.linear_model.LogisticRegression', {'multi_class':"multinomial",'solver':"lbfgs"} )

print_report('sklearn.linear_model.LogisticRegressionCV', {'multi_class':"multinomial",'solver':"lbfgs"} )

print_report('sklearn.neural_network.MLPClassifier')

print_report('sklearn.neighbors.NearestCentroid')

print_report('sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis')

#print_report('sklearn.neighbors.RadiusNeighborsClassifier',{'radius'=1.2})

print_report('sklearn.ensemble.RandomForestClassifier')

print_report('sklearn.linear_model.RidgeClassifier')



# Multiclass as One-Vs-One:

print_report('sklearn.svm.NuSVC')

print_report('sklearn.svm.SVC')

print_report('sklearn.gaussian_process.GaussianProcessClassifier', {'multi_class':"one_vs_one"})



# Multiclass as One-Vs-All

print_report('sklearn.ensemble.GradientBoostingClassifier')

print_report('sklearn.gaussian_process.GaussianProcessClassifier', {'multi_class':"one_vs_rest"})

print_report('sklearn.svm.LinearSVC', {'multi_class':"ovr"})

print_report('sklearn.linear_model.LogisticRegression', {'multi_class':"ovr"})

print_report('sklearn.linear_model.LogisticRegressionCV', {'multi_class':"ovr"})

print_report('sklearn.linear_model.SGDClassifier')

print_report('sklearn.linear_model.Perceptron')

print_report('sklearn.linear_model.PassiveAggressiveClassifier')