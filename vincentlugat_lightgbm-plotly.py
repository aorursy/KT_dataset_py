# Python libraries

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns

from datetime import datetime

import lightgbm as lgbm

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

import warnings

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff



warnings.filterwarnings('ignore')



from contextlib import contextmanager



@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))
#Read

data = pd.read_csv('../input/creditcard.csv')
display(data.head())

display(data.describe())

display(data.shape)

display(data.info())
plt.style.use('ggplot') # Using ggplot2 style visuals 



f, ax = plt.subplots(figsize=(11, 15))



ax.set_facecolor('#fafafa')

ax.set(xlim=(-5, 5))

plt.ylabel('Variables')

plt.title("Overview Data Set")

ax = sns.boxplot(data = data.drop(columns=['Amount', 'Class', 'Time']), 

  orient = 'h', 

  palette = 'Set2')
fraud = data[(data['Class'] != 0)]

normal = data[(data['Class'] == 0)]



trace = go.Pie(labels = ['Normal', 'Fraud'], values = data['Class'].value_counts(), 

               textfont=dict(size=15), opacity = 0.8,

               marker=dict(colors=['lightskyblue','gold'], 

                           line=dict(color='#000000', width=1.5)))





layout = dict(title =  'Distribution of target variable')

           

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
# Def plot distribution

def plot_distribution(data_select) : 

    figsize =( 15, 8)

    sns.set_style("ticks")

    s = sns.FacetGrid(data, hue = 'Class',aspect = 2.5, palette ={0 : 'lime', 1 :'black'})

    s.map(sns.kdeplot, data_select, shade = True, alpha = 0.6)

    s.set(xlim=(data[data_select].min(), data[data_select].max()))

    s.add_legend()

    s.set_axis_labels(data_select, 'proportion')

    s.fig.suptitle(data_select)

    plt.show()
#plot_distribution('V1')

#plot_distribution('V2')

#plot_distribution('V3')

plot_distribution('V4')

#plot_distribution('V5')

#plot_distribution('V6')

#plot_distribution('V7')

#plot_distribution('V8')

plot_distribution('V9')

#plot_distribution('V10')

plot_distribution('V11')

plot_distribution('V12')

plot_distribution('V13')

#plot_distribution('V14')

#plot_distribution('V15')

#plot_distribution('V16')

#plot_distribution('V17')

#plot_distribution('V18')

plot_distribution('V19')

#plot_distribution('V20')

#plot_distribution('V21')

#plot_distribution('V22')

#plot_distribution('V23')

plot_distribution('V24')

#plot_distribution('V25')

plot_distribution('V26')

#plot_distribution('V27')

#plot_distribution('V28')
# Correlation matrix 

f, (ax1, ax2) = plt.subplots(1,2,figsize =( 18, 8))

corr = data.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap((data.loc[data['Class'] ==1]).corr(), vmax = .8, square=True, ax = ax1, cmap = 'afmhot', mask=mask);

ax1.set_title('Fraud')

sns.heatmap((data.loc[data['Class'] ==0]).corr(), vmax = .8, square=True, ax = ax2, cmap = 'YlGnBu', mask=mask);

ax2.set_title('Normal')

plt.show()
# Normalization Amount

from sklearn.preprocessing import StandardScaler

data['nAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
# Drop useless variables

data = data.drop(['Amount','Time'],axis=1)
# def X and Y

y = np.array(data.Class.tolist())

data = data.drop('Class', 1)

X = np.array(data.as_matrix())
# Train_test split

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = random_state, stratify = y)
def model_performance(model) : 

    #Conf matrix

    conf_matrix = confusion_matrix(y_test, y_pred)

    trace1 = go.Heatmap(z = conf_matrix  ,x = ["0 (pred)","1 (pred)"],

                        y = ["0 (true)","1 (true)"],xgap = 2, ygap = 2, 

                        colorscale = 'Viridis', showscale  = False)



    #Show metrics

    tp = conf_matrix[1,1]

    fn = conf_matrix[1,0]

    fp = conf_matrix[0,1]

    tn = conf_matrix[0,0]

    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))

    Precision =  (tp/(tp+fp))

    Recall    =  (tp/(tp+fn))

    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))



    show_metrics = pd.DataFrame(data=[[Accuracy , Precision, Recall, F1_score]])

    show_metrics = show_metrics.T



    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']

    trace2 = go.Bar(x = (show_metrics[0].values), 

                   y = ['Accuracy', 'Precision', 'Recall', 'F1_score'], text = np.round_(show_metrics[0].values,4),

                    textposition = 'auto',

                   orientation = 'h', opacity = 0.8,marker=dict(

            color=colors,

            line=dict(color='#000000',width=1.5)))

    

    #Roc curve

    model_roc_auc = round(roc_auc_score(y_test, y_score) , 3)

    fpr, tpr, t = roc_curve(y_test, y_score)

    trace3 = go.Scatter(x = fpr,y = tpr,

                        name = "Roc : " + str(model_roc_auc),

                        line = dict(color = ('rgb(22, 96, 167)'),width = 2), fill='tozeroy')

    trace4 = go.Scatter(x = [0,1],y = [0,1],

                        line = dict(color = ('black'),width = 1.5,

                        dash = 'dot'))

    

    # Precision-recall curve

    precision, recall, thresholds = precision_recall_curve(y_test, y_score)

    trace5 = go.Scatter(x = recall, y = precision,

                        name = "Precision" + str(precision),

                        line = dict(color = ('lightcoral'),width = 2), fill='tozeroy')

    

    #Feature importance

    coefficients  = pd.DataFrame(eval(model).feature_importances_)

    column_data   = pd.DataFrame(list(data))

    coef_sumry    = (pd.merge(coefficients,column_data,left_index= True,

                              right_index= True, how = "left"))

    coef_sumry.columns = ["coefficients","features"]

    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

    coef_sumry = coef_sumry[coef_sumry["coefficients"] !=0]

    trace6 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],

                    name = "coefficients",

                    marker = dict(color = coef_sumry["coefficients"],

                                  colorscale = "Viridis",

                                  line = dict(width = .6,color = "black")))

    

    #Cumulative gain

    pos = pd.get_dummies(y_test).as_matrix()

    pos = pos[:,1] 

    npos = np.sum(pos)

    index = np.argsort(y_score) 

    index = index[::-1] 

    sort_pos = pos[index]

    #cumulative sum

    cpos = np.cumsum(sort_pos) 

    #recall

    recall = cpos/npos 

    #size obs test

    n = y_test.shape[0] 

    size = np.arange(start=1,stop=369,step=1) 

    #proportion

    size = size / n 

    #plots

    model = model

    trace7 = go.Scatter(x = size,y = recall,

                        name = "Lift curve",

                        line = dict(color = ('gold'),width = 2), fill='tozeroy') 

    

    #Subplots

    fig = tls.make_subplots(rows=4, cols=2, print_grid=False, 

                          specs=[[{}, {}], 

                                 [{}, {}],

                                 [{'colspan': 2}, None],

                                 [{'colspan': 2}, None]],

                          subplot_titles=('Confusion Matrix',

                                        'Metrics',

                                        'ROC curve'+" "+ '('+ str(model_roc_auc)+')',

                                        'Precision - Recall curve',

                                        'Cumulative gains curve',

                                        'Feature importance',

                                        ))

    

    fig.append_trace(trace1,1,1)

    fig.append_trace(trace2,1,2)

    fig.append_trace(trace3,2,1)

    fig.append_trace(trace4,2,1)

    fig.append_trace(trace5,2,2)

    fig.append_trace(trace6,4,1)

    fig.append_trace(trace7,3,1)

    

    fig['layout'].update(showlegend = False, title = '<b>Model performance report</b><br>'+str(model),

                        autosize = False, height = 1500,width = 830,

                        plot_bgcolor = 'rgba(240,240,240, 0.95)',

                        paper_bgcolor = 'rgba(240,240,240, 0.95)',

                        margin = dict(b = 195))

    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))

    fig["layout"]["xaxis3"].update(dict(title = "false positive rate"))

    fig["layout"]["yaxis3"].update(dict(title = "true positive rate"))

    fig["layout"]["xaxis4"].update(dict(title = "recall"), range = [0,1.05])

    fig["layout"]["yaxis4"].update(dict(title = "precision"), range = [0,1.05])

    fig["layout"]["xaxis5"].update(dict(title = "Percentage contacted"))

    fig["layout"]["yaxis5"].update(dict(title = "Percentage positive targeted"))

    fig.layout.titlefont.size = 14

    

    py.iplot(fig)
%%time

lgbm_clf = lgbm.LGBMClassifier(n_estimators=100, random_state = 42)



lgbm_clf.fit(X_train, y_train)

lgbm_clf.fit(X_train, y_train)

y_pred = lgbm_clf.predict(X_test)

y_score = lgbm_clf.predict_proba(X_test)[:,1]
model_performance('lgbm_clf')
fit_params = {"early_stopping_rounds" : 50, 

             "eval_metric" : 'binary', 

             "eval_set" : [(X_test,y_test)],

             'eval_names': ['valid'],

             'verbose': 0,

             'categorical_feature': 'auto'}



param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

              'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 5000],

              'num_leaves': sp_randint(6, 50), 

              'min_child_samples': sp_randint(100, 500), 

              'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

              'subsample': sp_uniform(loc=0.2, scale=0.8), 

              'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],

              'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

              'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

              'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}



#number of combinations

n_iter = 2 #(replace 2 by 200, 90 minutes)



#intialize lgbm and lunch the search

lgbm_clf = lgbm.LGBMClassifier(random_state=random_state, silent=True, metric='None', n_jobs=4)

grid_search = RandomizedSearchCV(

    estimator=lgbm_clf, param_distributions=param_test, 

    n_iter=n_iter,

    scoring='accuracy',

    cv=5,

    refit=True,

    random_state=random_state,

    verbose=True)



grid_search.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(grid_search.best_score_, grid_search.best_params_))



opt_parameters =  grid_search.best_params_



clf_sw = lgbm.LGBMClassifier(**lgbm_clf.get_params())

#Optimal parameter

clf_sw.set_params(**opt_parameters)
%%time

lgbm_clf = lgbm.LGBMClassifier(boosting_type='gbdt', class_weight=None,

        colsample_bytree=0.5112837457460335, importance_type='split',

        learning_rate=0.02, max_depth=7, metric='None',

        min_child_samples=195, min_child_weight=0.01, min_split_gain=0.0,

        n_estimators=3000, n_jobs=4, num_leaves=44, objective=None,

        random_state=42, reg_alpha=2, reg_lambda=10, silent=True,

        subsample=0.8137506311449016, subsample_for_bin=200000,

        subsample_freq=0)



lgbm_clf.fit(X_train, y_train)

lgbm_clf.fit(X_train, y_train)

y_pred = lgbm_clf.predict(X_test)

y_score = lgbm_clf.predict_proba(X_test)[:,1]
model_performance('lgbm_clf')
scores = cross_val_score(lgbm_clf, X, y, scoring = 'f1', cv=5)

trace = go.Table(

    header=dict(values=['<b>F1 score mean<b>', '<b>F1 score std<b>'],

                line = dict(color='#7D7F80'),

                fill = dict(color='#a1c3d1'),

                align = ['center'],

                font = dict(size = 15)),

    cells=dict(values=[np.round(scores.mean(),6),

                       np.round(scores.std(),6)],

               line = dict(color='#7D7F80'),

               fill = dict(color='#EDFAFF'),

               align = ['center'], font = dict(size = 15)))



layout = dict(width=800, height=500, title = 'Cross validation - 5 folds [F1 score]', font = dict(size = 15))

fig = dict(data=[trace], layout=layout)

py.iplot(fig, filename = 'styled_table')