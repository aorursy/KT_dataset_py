import pandas as pd

import numpy as np





import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline  



from sklearn import metrics



from sklearn import model_selection



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB







# load training data prepared previously



df_train = pd.read_csv('../input/train1.csv')

df_train.head()
# load test data prepared previously



df_test = pd.read_csv('../input/test1.csv')

df_test.head()
# original features

features_orig = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']



# original + extracted fetures

features_extr = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 'av1', 'av2', 'av3', 'av4', 'av5', 'av6', 'av7', 'av8', 'av9', 'av10', 'av11', 'av12', 'av13', 'av14', 'av15', 'av16', 'av17', 'av18', 'av19', 'av20', 'av21', 'sd1', 'sd2', 'sd3', 'sd4', 'sd5', 'sd6', 'sd7', 'sd8', 'sd9', 'sd10', 'sd11', 'sd12', 'sd13', 'sd14', 'sd15', 'sd16', 'sd17', 'sd18', 'sd19', 'sd20', 'sd21']



y_train = df_train['label_bnc']

y_test = df_test['label_bnc']



def bin_classify(model, clf, features, params=None, score=None):

    

    """Perfor Grid Search hyper parameter tuning on a classifier.

    

    Args:

        model (str): The model name identifier

        clf (clssifier object): The classifier to be tuned

        features (list): The set of input features names

        params (dict): Grid Search parameters

        score (str): Grid Search score

        

    Returns:

        Tuned Clssifier object

        dataframe of model predictions and scores

    

    """

    

    

    X_train = df_train[features]



    X_test = df_test[features] 

    

    grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring=score, n_jobs=-1)



    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)

    

    if hasattr(grid_search, 'predict_proba'):   

        y_score = grid_search.predict_proba(X_test)[:,1]

    elif hasattr(grid_search, 'decision_function'):

        y_score = grid_search.decision_function(X_test)

    else:

        y_score = y_pred

        

    predictions = {'y_pred' : y_pred, 'y_score' : y_score}

    df_predictions = pd.DataFrame.from_dict(predictions)

    

    return grid_search.best_estimator_, df_predictions

    
def bin_class_metrics(model, y_test, y_pred, y_score, print_out=True, plot_out=True):

    

    """Calculate main binary classifcation metrics, plot AUC ROC and Precision-Recall curves.

    

    Args:

        model (str): The model name identifier

        y_test (series): Contains the test label values

        y_pred (series): Contains the predicted values

        y_score (series): Contains the predicted scores

        print_out (bool): Print the classification metrics and thresholds values

        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves

        

    Returns:

        dataframe: The combined metrics in single dataframe

        dataframe: ROC thresholds

        dataframe: Precision-Recall thresholds

        Plot: AUC ROC

        plot: Precision-Recall

        plot: Precision-Recall threshold; also show the number of engines predicted for maintenace per period (queue).

        plot: TPR-FPR threshold

        

    """

      

    binclass_metrics = {

                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),

                        'Precision' : metrics.precision_score(y_test, y_pred),

                        'Recall' : metrics.recall_score(y_test, y_pred),

                        'F1 Score' : metrics.f1_score(y_test, y_pred),

                        'ROC AUC' : metrics.roc_auc_score(y_test, y_score)

                       }



    df_metrics = pd.DataFrame.from_dict(binclass_metrics, orient='index')

    df_metrics.columns = [model]  





    fpr, tpr, thresh_roc = metrics.roc_curve(y_test, y_score)

    

    roc_auc = metrics.auc(fpr, tpr)



    engines_roc = []  

    for thr in thresh_roc:  

        engines_roc.append((y_score >= thr).mean())



    engines_roc = np.array(engines_roc)



    roc_thresh = {

                    'Threshold' : thresh_roc,

                    'TPR' : tpr,

                    'FPR' : fpr,

                    'Que' : engines_roc

                 }

    

    df_roc_thresh = pd.DataFrame.from_dict(roc_thresh)

    

    #calculate other classification metrics: TP, FP, TN, FN, TNR, FNR

    #from ground truth file, positive class = 25 => TP + FN = 25

    #from ground truth file, negative class = 75 => TN + FP = 75

    

    df_roc_thresh['TP'] = (25*df_roc_thresh.TPR).astype(int)

    df_roc_thresh['FP'] = (25 - (25*df_roc_thresh.TPR)).astype(int)

    df_roc_thresh['TN'] = (75*(1 - df_roc_thresh.FPR)).astype(int)

    df_roc_thresh['FN'] = (75 - (75*(1 - df_roc_thresh.FPR))).astype(int)

    

    df_roc_thresh['TNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FN'])

    df_roc_thresh['FNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FP'])

    

    df_roc_thresh['Model'] = model



    



    precision, recall, thresh_prc = metrics.precision_recall_curve(y_test, y_score)



    thresh_prc = np.append(thresh_prc,1)



    engines_prc = []  

    for thr in thresh_prc:  

        engines_prc.append((y_score >= thr).mean())



    engines_prc = np.array(engines_prc)



    prc_thresh = {

                    'Threshold' : thresh_prc,

                    'Precision' : precision,

                    'Recall' : recall,

                    'Que' : engines_prc

                 }



    df_prc_thresh = pd.DataFrame.from_dict(prc_thresh)



    if print_out:

        print('-----------------------------------------------------------')

        print(model, '\n')

        print('Confusion Matrix:')

        print(metrics.confusion_matrix(y_test, y_pred))

        print('\nClassification Report:')

        print(metrics.classification_report(y_test, y_pred))

        print('\nMetrics:')

        print(df_metrics)



        print('\nROC Thresholds:\n')

        print(df_roc_thresh[['Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'TNR','FNR', 'Que']])



        print('\nPrecision-Recall Thresholds:\n')

        print(df_prc_thresh[['Threshold', 'Precision', 'Recall', 'Que']])



    if plot_out:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False )

        fig.set_size_inches(10,10)



        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f'% roc_auc)

        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        ax1.set_xlim([-0.05, 1.0])

        ax1.set_ylim([0.0, 1.05])

        ax1.set_xlabel('False Positive Rate')

        ax1.set_ylabel('True Positive Rate')

        ax1.legend(loc="lower right", fontsize='small')



        ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')

        ax2.set_xlim([0.0, 1.0])

        ax2.set_ylim([0.0, 1.05])

        ax2.set_xlabel('Recall')

        ax2.set_ylabel('Precision')

        ax2.legend(loc="lower left", fontsize='small')



        ax3.plot(thresh_roc, fpr, color='red', lw=2, label='FPR')  

        ax3.plot(thresh_roc, tpr, color='green',label='TPR') 

        ax3.plot(thresh_roc, engines_roc, color='blue',label='Engines') 

        ax3.set_ylim([0.0, 1.05])

        ax3.set_xlabel('Threshold')  

        ax3.set_ylabel('%')

        ax3.legend(loc='upper right', fontsize='small')



        ax4.plot(thresh_prc, precision, color='red', lw=2, label='Precision')  

        ax4.plot(thresh_prc, recall, color='green',label='Recall') 

        ax4.plot(thresh_prc, engines_prc, color='blue',label='Engines') 

        ax4.set_ylim([0.0, 1.05])

        ax4.set_xlabel('Threshold')  

        ax4.set_ylabel('%')

        ax4.legend(loc='lower left', fontsize='small')



    return  df_metrics, df_roc_thresh, df_prc_thresh

model = 'Logistic Regression B'

clf_lgrb = LogisticRegression(random_state=123)

gs_params = {'C': [.01, 0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}

gs_score = 'roc_auc'



clf_lgrb, pred_lgrb = bin_classify(model, clf_lgrb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_lgrb)



metrics_lgrb, roc_lgrb, prc_lgrb = bin_class_metrics(model, y_test, pred_lgrb.y_pred, pred_lgrb.y_score, print_out=True, plot_out=True)
model = 'Logistic Regression A'

clf_lgra = LogisticRegression(random_state=123)

gs_params = {'C': [.01, 0.1, 1.0, 10], 'solver': ['liblinear', 'lbfgs']}

gs_score = 'roc_auc'



clf_lgra, pred_lgra = bin_classify(model, clf_lgra, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_lgra)



metrics_lgra, roc_lgra, prc_lgra = bin_class_metrics(model, y_test, pred_lgra.y_pred, pred_lgra.y_score, print_out=True, plot_out=True)
metrics_lgr = pd.concat([metrics_lgrb, metrics_lgra], axis=1)

metrics_lgr
model = 'Decision Tree B'

clf_dtrb = DecisionTreeClassifier(random_state=123)

gs_params = {'max_depth': [2, 3, 4, 5, 6], 'criterion': ['gini', 'entropy']}

gs_score = 'roc_auc'



clf_dtrb, pred_dtrb = bin_classify(model, clf_dtrb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_dtrb)



metrics_dtrb, roc_dtrb, prc_dtrb = bin_class_metrics(model, y_test, pred_dtrb.y_pred, pred_dtrb.y_score, print_out=True, plot_out=True)
model = 'Decision Tree A'

clf_dtra = DecisionTreeClassifier(random_state=123)

gs_params = {'max_depth': [3, 4, 5, 6, 7], 'criterion': ['gini', 'entropy']}

gs_score = 'roc_auc'



clf_dtra, pred_dtra = bin_classify(model, clf_dtra, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_dtra)



metrics_dtra, roc_dtra, prc_dtra = bin_class_metrics(model, y_test, pred_dtra.y_pred, pred_dtra.y_score, print_out=True, plot_out=True)

metrics_dtr = pd.concat([metrics_dtrb, metrics_dtra], axis=1)

metrics_dtr
model = 'Random Forest B'

clf_rfcb = RandomForestClassifier(n_estimators=50, random_state=123)

gs_params = {'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}

gs_score = 'roc_auc'



clf_rfcb, pred_rfcb = bin_classify(model, clf_rfcb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_rfcb)



metrics_rfcb, roc_rfcb, prc_rfcb = bin_class_metrics(model, y_test, pred_rfcb.y_pred, pred_rfcb.y_score, print_out=True, plot_out=True)
model = 'Random Forest A'

clf_rfca = RandomForestClassifier(n_estimators=50, random_state=123)

gs_params = {'max_depth': [4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy']}

gs_score = 'roc_auc'



clf_rfca, pred_rfca = bin_classify(model, clf_rfca, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_rfca)



metrics_rfca, roc_rfca, prc_rfca = bin_class_metrics(model, y_test, pred_rfca.y_pred, pred_rfca.y_score, print_out=True, plot_out=True)
metrics_rfc = pd.concat([metrics_rfcb, metrics_rfca], axis=1)

metrics_rfc
model = 'SVC B'

clf_svcb = SVC(kernel='rbf', random_state=123)

gs_params = {'C': [1.0]}

gs_score = 'roc_auc'



clf_svcb, pred_svcb = bin_classify(model, clf_svcb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_svcb)



metrics_svcb, roc_svcb, prc_svcb = bin_class_metrics(model, y_test, pred_svcb.y_pred, pred_svcb.y_score, print_out=True, plot_out=True)
model = 'SVC A'

clf_svca = SVC(kernel='rbf', random_state=123)

gs_params = {'C': [1.0]}

gs_score = 'roc_auc'



clf_svca, pred_svca = bin_classify(model, clf_svca, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_svca)



metrics_svca, roc_svca, prc_svca = bin_class_metrics(model, y_test, pred_svca.y_pred, pred_svca.y_score, print_out=True, plot_out=True)
# Compare SVC Before and After FE

metrics_svc = pd.concat([metrics_svcb, metrics_svca], axis=1)

metrics_svc.sort_index()



model = 'SVC Linear B'

clf_svlb = LinearSVC(random_state=123)

gs_params = {'C': [.01 ,.1 ,1.0]}

gs_score = 'roc_auc'



clf_svlb, pred_svlb = bin_classify(model, clf_svlb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_svlb)



metrics_svlb, roc_svlb, prc_svlb = bin_class_metrics(model, y_test, pred_svlb.y_pred, pred_svlb.y_score, print_out=True, plot_out=True)
model = 'SVC Linear A'

clf_svla = LinearSVC(random_state=123)

gs_params = {'C': [.001, .01 ,.1 ]}

gs_score = 'roc_auc'



clf_svla, pred_svla = bin_classify(model, clf_svla, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_svla)



metrics_svla, roc_svla, prc_svla = bin_class_metrics(model, y_test, pred_svla.y_pred, pred_svla.y_score, print_out=True, plot_out=True)



# Compare Linear SVC Before and After FE

metrics_svl = pd.concat([metrics_svlb, metrics_svla], axis=1)

metrics_svl.sort_index()
model = 'KNN B'

clf_knnb = KNeighborsClassifier(n_jobs=-1)

gs_params = {'n_neighbors': [9, 10, 11, 12, 13]}

gs_score = 'roc_auc'



clf_knnb, pred_knnb = bin_classify(model, clf_knnb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_knnb)



metrics_knnb, roc_knnb, prc_knnb = bin_class_metrics(model, y_test, pred_knnb.y_pred, pred_knnb.y_score, print_out=True, plot_out=True)
model = 'KNN A'

clf_knna = KNeighborsClassifier(n_jobs=-1)

gs_params = {'n_neighbors': [9 , 10, 11, 12, 13]}

gs_score = 'roc_auc'



clf_knna, pred_knna = bin_classify(model, clf_knna, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_knna)



metrics_knna, roc_knna, prc_knna = bin_class_metrics(model, y_test, pred_knna.y_pred, pred_knna.y_score, print_out=True, plot_out=True)
# Compare KNN Before and After FE

metrics_knn = pd.concat([metrics_knnb, metrics_knna], axis=1)

metrics_knn.sort_index()
model = 'Gaussian NB B'

clf_gnbb = GaussianNB()

gs_params = {} 

gs_score = 'roc_auc'



clf_gnbb, pred_gnbb = bin_classify(model, clf_gnbb, features_orig, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_gnbb)



metrics_gnbb, roc_gnbb, prc_gnbb = bin_class_metrics(model, y_test, pred_gnbb.y_pred, pred_gnbb.y_score, print_out=True, plot_out=True)
model = 'Gaussian NB A'

clf_gnba = GaussianNB()

gs_params = {} 

gs_score = 'roc_auc'



clf_gnba, pred_gnba = bin_classify(model, clf_gnba, features_extr, params=gs_params, score=gs_score)

print('\nBest Parameters:\n',clf_gnba)



metrics_gnba, roc_gnba, prc_gnba = bin_class_metrics(model, y_test, pred_gnba.y_pred, pred_gnba.y_score, print_out=True, plot_out=True)

# Compare Naive Bayes Before and After FE

metrics_gnb = pd.concat([metrics_gnbb, metrics_gnba], axis=1)

metrics_gnb.sort_index()
#compare all models

metrics_bn = pd.concat([metrics_lgr, metrics_dtr, metrics_rfc, metrics_svc, metrics_svl, metrics_knn, metrics_gnb], axis=1)

metrics_bn
# Plot AUC-ROC and precision-recall curves for best models



fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False )

fig.set_size_inches(10,5)



ax1.plot(roc_gnbb.FPR, roc_gnbb.TPR, color='yellow', lw=1, label= metrics_gnbb.columns.values.tolist()[0] + ':  %.3f' % metrics_gnbb.at['ROC AUC', metrics_gnbb.columns.values.tolist()[0]])

ax1.plot(roc_rfca.FPR, roc_rfca.TPR, color='green', lw=1, label= metrics_rfca.columns.values.tolist()[0] + ':  %.3f' % metrics_rfca.at['ROC AUC', metrics_rfca.columns.values.tolist()[0]])

ax1.plot(roc_lgra.FPR, roc_lgra.TPR, color='blue', lw=1, label= metrics_lgra.columns.values.tolist()[0] + ':  %.3f' % metrics_lgra.at['ROC AUC', metrics_lgra.columns.values.tolist()[0]])

ax1.plot(roc_svla.FPR, roc_svla.TPR, color='brown', lw=1, label= metrics_svla.columns.values.tolist()[0] + ':  %.3f' % metrics_svla.at['ROC AUC', metrics_svla.columns.values.tolist()[0]])

ax1.plot(roc_svlb.FPR, roc_svlb.TPR, color='sandybrown', lw=1, label= metrics_svlb.columns.values.tolist()[0] + ':  %.3f' % metrics_svlb.at['ROC AUC', metrics_svlb.columns.values.tolist()[0]])

ax1.plot(roc_knna.FPR, roc_knna.TPR, color='darkmagenta', lw=1, label= metrics_knna.columns.values.tolist()[0] + ':  %.3f' % metrics_knna.at['ROC AUC', metrics_knna.columns.values.tolist()[0]])

ax1.plot(roc_svca.FPR, roc_svca.TPR, color='red', lw=1, label= metrics_svca.columns.values.tolist()[0] + ':  %.3f' % metrics_svca.at['ROC AUC', metrics_svca.columns.values.tolist()[0]])

ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

ax1.set_xlim([-0.05, 1.0])

ax1.set_ylim([0.0, 1.05])

ax1.set_xlabel('False Positive Rate')

ax1.set_ylabel('True Positive Rate')

ax1.legend(loc="lower right", fontsize='small')

ax1.set_title('AUC ROC')



ax2.plot(prc_gnbb.Recall, prc_gnbb.Precision, color='yellow', lw=1, label= metrics_gnbb.columns.values.tolist()[0])

ax2.plot(prc_rfca.Recall, prc_rfca.Precision, color='green', lw=1, label= metrics_rfca.columns.values.tolist()[0])

ax2.plot(prc_lgra.Recall, prc_lgra.Precision, color='blue', lw=1, label= metrics_lgra.columns.values.tolist()[0])

ax2.plot(prc_svla.Recall, prc_svla.Precision, color='brown', lw=1, label= metrics_svla.columns.values.tolist()[0])

ax2.plot(prc_svlb.Recall, prc_svlb.Precision, color='sandybrown', lw=1, label= metrics_svlb.columns.values.tolist()[0])

ax2.plot(prc_knna.Recall, prc_knna.Precision, color='darkmagenta', lw=1, label= metrics_knna.columns.values.tolist()[0])

ax2.plot(prc_svca.Recall, prc_svca.Precision, color='red', lw=1, label= metrics_svca.columns.values.tolist()[0])

ax2.set_xlim([0.0, 1.0])

ax2.set_ylim([0.0, 1.05])

ax2.set_xlabel('Recall')

ax2.set_ylabel('Precision')

ax2.legend(loc="lower left", fontsize='small')

ax2.set_title('Precision Recall Curve')

        
TPb = 300

TNb = 0

FPc = -100

FNc = -200



prob_positive = 0.25 

prop_negative = 0.75

# calculate the max profit in each model using roc dataframes generated earlier during model training



df_max_profit = pd.DataFrame()



roc_dfs = [roc_lgrb, roc_lgra, roc_dtrb, roc_dtra, roc_rfcb, roc_rfca, roc_svcb, roc_svca, roc_svlb, roc_svla, roc_knnb, roc_knna, roc_gnbb, roc_gnba]



for df in roc_dfs:

    df['Profit'] = prob_positive*(df.TPR*TPb + df.FNR*FNc) + prop_negative*(df.TNR*TNb + df.FPR*FPc)

    df_max = df[df.Profit == df.Profit.max()]

    df_max_profit = pd.concat([df_max_profit,df_max], ignore_index=True)

    

colnames = ['Profit', 'Model', 'Que', 'Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'TNR', 'FNR']



# sort results by Profit

df_max_profit = df_max_profit[colnames].sort_values(by='Profit', ascending=False).reset_index(drop=True)

df_max_profit

# sort the results by Que (capacity) and Profit



df_max_profit = df_max_profit[colnames].sort_values(by=['Que', 'Profit'], ascending=[True, False]).reset_index(drop=True)

df_max_profit
