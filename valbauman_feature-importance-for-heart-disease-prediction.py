import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score



#load data and create 90/10 train/validation splits

data= pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv', delimiter= ',')

top5_feats= ['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']

feats= data.loc[:,top5_feats] #only keep 5 of the 12 feature columns

labels= data.iloc[:,-1]

x_train, x_devel, y_train, y_devel= train_test_split(feats, labels, test_size= 0.1, random_state= 20)



#scale features (0 mean, unit variance for each)

scaler= StandardScaler()

x_train_scale= scaler.fit_transform(x_train)

x_devel_scale= scaler.transform(x_devel)



#train SVM

svm= SVC(probability= True)

svm.fit(x_train_scale, y_train)



#for interest's sake, we'll display the average recall of each class for this model

predicts= svm.predict(x_devel_scale)

print(balanced_accuracy_score(y_devel, predicts))
def create_hist(feature_vals, labels):

    """

    FUNCTION TO DISPLAY HISTOGRAMS OF THE FEATURE VALUES FOR THE TWO CLASSES

    

    INPUTS:

    feature_vals : pd series of the values of the single feature of interest

    labels : pd series of the corresponding labels for the feature values

    

    OUTPUTS:

    None (displays histograms only)

    """

    feat_0= feature_vals.loc[y_train == 0] #feature values that belong to class 0

    feat_1= feature_vals.loc[y_train == 1] #features values that belong to class 1

    

    #for histogram bins, get the min and max feature values. we'll use a total of 10 bins using the min/max

    feat_min= np.min(feature_vals)

    feat_max= np.max(feature_vals)

    bins= np.linspace(feat_min, feat_max, num= 10)

    

    #display histograms. illustrates the distribution of values for the feature for each class

    plt.figure()

    plt.hist(feat_0, bins, alpha= 0.5, label= 'Target= 0')

    plt.hist(feat_1, bins, alpha= 0.5, label= 'Target= 1')

    plt.legend()

    plt.xlabel('Feature value'); plt.ylabel('Frequency')

    plt.title(('Feature value histograms - '+ feature_vals.name))

    

    return



for i in top5_feats:

    create_hist(x_train[i], y_train)
from sklearn.inspection import permutation_importance



#express scaled development set as a dataframe before getting importances

x_devel_scaledf= pd.DataFrame(x_devel_scale, columns= x_devel.columns, index= x_devel.index)



permut_importance= permutation_importance(svm, x_devel_scaledf, y_devel, n_repeats= 5, random_state= 20)

print(x_devel_scaledf.columns)

print(permut_importance['importances_mean'].round(3))

print(permut_importance['importances_std'].round(3))
import shap



#explain the model and then get SHAP values

explain= shap.KernelExplainer(svm.predict_proba, x_train_scale, link= 'logit')

shap_vals= explain.shap_values(x_devel_scale, nsamples= len(x_devel_scale), l1_reg= 'num_features(5)')



#get shap summary plot

shap.summary_plot(shap_vals, x_devel_scaledf)
shap.summary_plot(shap_vals[0],x_devel_scaledf)
shap.dependence_plot('time', shap_vals[0], x_devel_scaledf, interaction_index= 'ejection_fraction')