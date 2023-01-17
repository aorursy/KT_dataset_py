import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/advertising/advertising.csv')

data.head()
from sklearn import metrics

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.preprocessing import StandardScaler
features = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1)

target = data['Clicked on Ad']

ss = StandardScaler()



log = LogisticRegression()

dec = DecisionTreeClassifier()

ran = RandomForestClassifier()

nn = MLPClassifier()



log_pipe = make_pipeline(ss,log)

dec_pipe = make_pipeline(ss,dec)

ran_pipe = make_pipeline(ss,ran)

nn_pipe = make_pipeline(ss,nn)



kf = KFold(n_splits=10, shuffle=True, random_state = 1)



log_results = cross_val_score(log_pipe, features, target, cv=kf, scoring = 'accuracy')

dec_results = cross_val_score(dec_pipe, features, target, cv=kf, scoring = 'accuracy')

ran_results = cross_val_score(ran_pipe, features, target, cv=kf, scoring = 'accuracy')

nn_results = cross_val_score(nn_pipe, features, target, cv=kf, scoring = 'accuracy')



pd.DataFrame({'Algorithm':['Logistic Regression','Decision Tree','Random Forest','Neural Network'],

             'K-Fold Accuracy':[log_results.mean(),dec_results.mean(),

                               ran_results.mean(),nn_results.mean()]})
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2)



log = LogisticRegression()

dec = DecisionTreeClassifier()

ran = RandomForestClassifier()

nn = MLPClassifier()



models = [log,dec,ran,nn]



for model in models:

    model.fit(X_train,y_train)

    

for model in models:

    print(model)

    print(classification_report(y_test,model.predict(X_test)))

    print("")
model = ran
import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
from pdpbox import pdp, info_plots

import matplotlib.pyplot as plt



base_features = X_train.columns.tolist()



for feature in base_features:

    

    feat_name = feature

    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()
feature_list = X_test.columns.tolist()

feature_list.remove('Male')



start_index = 1

for feature in feature_list:

    for index in range(start_index,4):



        features = [feature,feature_list[index]]



        inter  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=base_features, features=features)



        pdp.pdp_interact_plot(pdp_interact_out=inter, feature_names=features, plot_type='contour')

        plt.show()

    

    start_index += 1