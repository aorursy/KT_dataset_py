import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import precision_score, recall_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn import metrics

from sklearn.metrics import classification_report

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE

from shapely.geometry import LineString

import warnings

warnings.filterwarnings("ignore")
bank_data = pd.read_csv("/kaggle/input/bank-term-deposit/bank-full.csv",sep = ";")

bank_data.head()
#rename the column 'y' to 'target'

bank_data.rename(columns={"y": "target"},inplace = True)
bank_data.shape
bank_data.info()
#Check if any entry is null

bank_data.isnull().sum()
bank_data.describe()
bank_data.columns
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,20])

cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

cnt = 1

for col in cols :

    ax = plt.subplot(4,2,cnt)

    sns.distplot(bank_data[col], hist_kws=dict(edgecolor="k", linewidth=1,color='grey'), color='red')

    cnt+=1

    plot_name = "Data distribution of column : "+col

    ax.set_title(plot_name,fontsize = 15)

plt.tight_layout()

plt.show() 
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,15])

cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

cnt = 1

for col in cols :

    ax = plt.subplot(4,2,cnt)

    sns.boxplot(bank_data[col])

    cnt+=1

    plot_name = "IQR for column : "+col

    ax.set_title(plot_name,fontsize = 15)

plt.tight_layout()

plt.show() 
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,15])

cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

cnt = 1

for col in cols :

    ax = plt.subplot(3,2,cnt)

    sns.violinplot(data = bank_data,x = col, y='target')

    cnt+=1

    plot_title = "Data distribution of "+col+" for output labels"

    ax.set_title(plot_title,fontsize = 15)

plt.tight_layout()

plt.show() 
g = sns.pairplot(data = bank_data,hue = "target",diag_kws={'bw': 0.2})

g.fig.suptitle("Pairplot of numerical columns in the dataset",y = 1)

plt.show()
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,20])

cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'job', 'target']

cnt = 1

for col in cols :

    ax = plt.subplot(5,2,cnt)

    sns.countplot(data = bank_data, x = col, order = bank_data[col].value_counts().index)

    if col == 'job' :

        plt.xticks(rotation = 90)

    cnt+=1

    plot_name = "Countplot for column : "+col

    ax.set_title(plot_name,fontsize = 15)

plt.tight_layout()

plt.show()  
bank_data.target.value_counts()
bank_data.target.value_counts().plot(kind = 'pie', autopct='%.2f')

plt.title("Target variable percentage",fontsize = 15)

plt.show()
bank_data_yes = bank_data[bank_data.target == 'yes']

bank_data_yes.head()
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,20])

cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

cnt = 1

for col in cols :

    ax = plt.subplot(4,2,cnt)

    sns.distplot(bank_data_yes[col], hist_kws=dict(edgecolor="k", linewidth=1,color='grey'), color='red')

    cnt+=1

    plot_name = "Data distribution of column : '"+col+"' for target label 1"

    ax.set_title(plot_name,fontsize = 15)

plt.tight_layout()

plt.show() 
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,15])

cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

cnt = 1

for col in cols :

    ax = plt.subplot(4,2,cnt)

    sns.boxplot(bank_data_yes[col])

    cnt+=1

    plot_name = "IQR for column : "+col

    ax.set_title(plot_name,fontsize = 15)

plt.tight_layout()

plt.show() 
sns.set_style("whitegrid")

fig = plt.figure(figsize = [15,20])

cols = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'job']

cnt = 1

for col in cols :

    ax = plt.subplot(5,2,cnt)

    sns.countplot(data = bank_data_yes, x = col, order = bank_data_yes[col].value_counts().index)

    if col == 'job' :

        plt.xticks(rotation = 90)

    plot_name = "Countplot for column : '"+col+"' for target label 1"

    ax.set_title(plot_name,fontsize = 15)

    cnt+=1

plt.tight_layout()

plt.show()  
# List of variables to map



varlist =  ['housing', 'loan', 'default', 'target']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# Applying the function to the housing list

bank_data[varlist] = bank_data[varlist].apply(binary_map)
bank_data.head()
def month_converter(month):

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    return months.index(month) + 1
bank_data.month = bank_data.month.apply(month_converter)

bank_data.head()
# correlation matrix 

plt.figure(figsize = (20,10))        

sns.heatmap(bank_data.corr(),annot = True, fmt='.2f')

plt.yticks(rotation=0) 

plt.show()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy = pd.get_dummies(bank_data[['marital', 'education', 'contact', 'poutcome', 'job']], drop_first=True)



# Adding the results to the master dataframe

bank_data = pd.concat([bank_data, dummy], axis=1)
bank_data.shape
# We have created dummies for the below variables, so we can drop them

bank_data = bank_data.drop(['marital', 'education', 'contact', 'poutcome', 'job'], 1)

bank_data.shape
df_train,df_test = train_test_split(bank_data, train_size=0.7, test_size=0.3, random_state=0)

df_train.to_csv('df_train.csv',index = None)
scaler = StandardScaler()

cols = ['age','balance','day','month','duration','campaign', 'pdays', 'previous']

bank_data[cols] = scaler.fit_transform(bank_data[cols])

bank_data.head()
bank_data.info()
def logReg(x,y) :

    X_train_sm = sm.add_constant(x)

    logm1 = sm.GLM(y,X_train_sm, family = sm.families.Binomial())

    res = logm1.fit()

    return res
def get_classification_report(res, y_df, X_df, prob) :

    y_df_pred = res.predict(X_df)#.values.reshape(-1)

    y_df_pred_final = pd.DataFrame({'Target':y_df.values, 'Target_Prob':y_df_pred})

    y_df_pred_final['predicted'] = y_df_pred_final.Target_Prob.map(lambda x: 1 if x > prob else 0)

    #print(classification_report(y_df_pred_final.Target, y_df_pred_final.predicted))

    return y_df_pred_final
def precision_recall(actual,predicted) :

    recall = round(recall_score(actual,predicted),2)

    print("precision : ",round(precision_score(actual,predicted),2))

    print("recall : ",recall)

    return recall
def get_confusion_matrix(actual,predicted) :

    cm = confusion_matrix(actual,predicted)

    df = pd.DataFrame(cm)

    return df
df = bank_data.copy()

X = df.drop('target',axis = 1)

y = df['target']

X_aux, X_test, y_aux, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_aux, y_aux, train_size=0.75, test_size=0.25, random_state=0)
res = logReg(X_train,y_train)

res.summary()
def get_vif(X_train) :

    vif = pd.DataFrame()

    vif['Features'] = X_train.columns

    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return vif
get_vif(X_train)
X_train_new = X_train.drop(['poutcome_unknown'],axis = 1)

X_val_new = X_val.drop(['poutcome_unknown'],axis=1)

X_test_new = X_test.drop(['poutcome_unknown'],axis = 1)

res = logReg(X_train_new,y_train)
get_vif(X_train_new)
y_df_pred_final = get_classification_report(res,y_train,sm.add_constant(X_train_new),0.5)

precision_recall(y_df_pred_final.Target,y_df_pred_final.predicted)
y_df_pred_final
def output_for_probability_range(target, target_prob) :

    y_df_pred_final = pd.DataFrame({'Target':target, 'Target_Prob':target_prob})

    numbers = [float(x)/10 for x in range(10)]

    for i in numbers:

        y_df_pred_final[i]= y_df_pred_final.Target_Prob.map(lambda x: 1 if x > i else 0)

        

    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for i in num:

        cm1 = metrics.confusion_matrix(y_df_pred_final.Target, y_df_pred_final[i] )

        total1=sum(sum(cm1))

        accuracy = (cm1[0,0]+cm1[1,1])/total1

        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])

        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])

        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

    return cutoff_df
cutoff_df = output_for_probability_range(y_df_pred_final.Target,y_df_pred_final.Target_Prob)

cutoff_df
def get_optimal_threshold_curve(cutoff_df) :

    x = cutoff_df.prob

    f = cutoff_df.accuracy

    g = cutoff_df.sensi

    h = cutoff_df.speci



    plt.plot(x, f, '-',label = 'accuracy')

    plt.plot(x, g, '-',label = 'sensitivity')

    plt.plot(x, h, '-',label = 'specificity')



    first_line = LineString(np.column_stack((x, f)))

    second_line = LineString(np.column_stack((x, g)))

    intersection = first_line.intersection(second_line)

    plt.xticks(np.arange(0, 1, step=0.1))

    plt.plot(*intersection.xy, 'o')

    plt.legend()

    plt.grid()

    plt.savefig('optimal_probability_graph.png', bbox_inches='tight')

    plt.show()

    x,y = intersection.xy

    prob_threshold = list(x)[0]

    return prob_threshold
prob_threshold = get_optimal_threshold_curve(cutoff_df)

prob_threshold
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.grid()

    plt.legend(loc="lower right")

    plt.savefig('Receiver_Operating_Characteristic.png', bbox_inches='tight')

    plt.show()



    return None
y_train_pred_final = get_classification_report(res, y_train, sm.add_constant(X_train_new), prob_threshold)

draw_roc(y_train_pred_final.Target, y_train_pred_final.Target_Prob)
precision_recall(y_train_pred_final.Target, y_train_pred_final.predicted)
X_val_sm = sm.add_constant(X_val_new)

y_pred_lr = res.predict(X_val_sm)

y_df_pred_final = get_classification_report(res, y_val, X_val_sm, prob_threshold)
recall_lr = precision_recall(y_df_pred_final.Target, y_df_pred_final.predicted)
X_test_sm = sm.add_constant(X_test_new)

y_pred_lr = res.predict(X_test_sm)

y_df_pred_final = get_classification_report(res, y_test, X_test_sm, prob_threshold)
precision_recall(y_df_pred_final.Target, y_df_pred_final.predicted)
get_confusion_matrix(y_df_pred_final.Target, y_df_pred_final.predicted)
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}
elasticnet = ElasticNet()

folds = 5

# cross validation

model_cv = GridSearchCV(estimator = elasticnet, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(np.log10(cv_results['param_alpha']), cv_results['mean_train_score'])

plt.plot(np.log10(cv_results['param_alpha']), cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.grid()

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_estimator_
alpha = model_cv.best_estimator_.alpha

elasticnet = ElasticNet(alpha=alpha)       

elasticnet.fit(X_train, y_train) 

elasticnet.coef_
y_pred_reg = elasticnet.predict(X_test)

print("Mean squared error with LogReg : ",np.round(metrics.mean_squared_error(y_test,y_pred_lr),2))

print("Mean squared error with Regularization : ",np.round(metrics.mean_squared_error(y_test,y_pred_reg),2))
print("R2 score with LogReg : ",np.round(metrics.r2_score(y_test,y_pred_lr),2))

print("R2 score with Regularization : ",np.round(metrics.r2_score(y_test,y_pred_reg),2))
y_df_pred_final_reg = get_classification_report(elasticnet, y_test, X_test, prob_threshold)

precision_recall(y_df_pred_final_reg.Target, y_df_pred_final_reg.predicted)
get_confusion_matrix(y_df_pred_final_reg.Target, y_df_pred_final_reg.predicted)
# Create a Decision Tree

dt_basic = DecisionTreeClassifier(max_depth=10)

dt_basic
# Fit the training data

dt_basic.fit(X_train,y_train)
# Predict based on test data

y_preds = dt_basic.predict(X_val)
precision_recall(y_val,y_preds)
# Calculate the number of nodes in the tree

dt_basic.tree_.node_count
dt_basic.tree_.max_depth
# Create a Parameter grid

param_grid = {

    'classification__max_depth' : range(10,16),

    'classification__min_samples_leaf' : range(5,60,5),

    'classification__min_samples_split' : range(5,60,5),

    'classification__criterion' : ['gini','entropy'],

    'classification__max_features' : ['auto','sqrt','log2']

}
n_folds = 4
# Create a Decision Tree

dtree = DecisionTreeClassifier()
model = Pipeline([

        ('sampling', SMOTE()),

        ('classification', dtree)

    ])
# Create a Grid with parameters

grid = GridSearchCV(model, param_grid, cv = n_folds, n_jobs = -1,return_train_score=True,scoring = 'recall')
%%time

grid.fit(X_train,y_train)
scores = grid.cv_results_
cv_result = pd.DataFrame(scores)

cv_result.head()
# Plot accuracy vs param_max_depth

plt.figure

plt.plot(scores['param_classification__max_depth'].data,scores['mean_train_score'], label = "training_accuracy")

plt.plot(scores['param_classification__max_depth'].data,scores['mean_test_score'], label = "test_accuracy")

plt.xlabel("max_depth")

plt.ylabel("accuracy")

plt.legend()

plt.show()
grid.best_params_
best_grid = grid.best_estimator_

best_grid
best_grid.fit(X_train,y_train)
dtree_prob = best_grid.predict_proba(X_train)
cutoff_df = output_for_probability_range(y_train,dtree_prob[:,1])

cutoff_df
prob_threshold_dt = get_optimal_threshold_curve(cutoff_df)

prob_threshold_dt
def get_classification_report_tree(res, y_df, X_df, prob) :

    y_df_pred = res.predict_proba(X_df)[:,1]

    y_df_pred_final = pd.DataFrame({'Target':y_df.values, 'Target_Prob':y_df_pred})

    y_df_pred_final['predicted'] = y_df_pred_final.Target_Prob.map(lambda x: 1 if x > prob else 0)

    return y_df_pred_final
dt_y_train_pred_final = get_classification_report_tree(best_grid, y_train, X_train, prob_threshold_dt)

draw_roc(dt_y_train_pred_final.Target, dt_y_train_pred_final.Target_Prob)
y_val_pred = best_grid.predict(X_val)

recall_dt = precision_recall(y_val,y_val_pred)
y_preds = best_grid.predict(X_test)

precision_recall(y_test,y_preds)
get_confusion_matrix(y_test,y_preds)
rf = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=4)
rf.fit(X_train, y_train)
y_preds = rf.predict(X_val)

precision_recall(y_val,y_preds)
X_train.shape
cols = X_train.columns

len(cols)
# Create the parameter grid based on the results of random search 

params = {

    'classification__max_depth': [7,8,10],

    'classification__min_samples_leaf': range(50,100,10),

    'classification__max_features': range(4,12,2),

    'classification__n_estimators': range(20,100,20),

    'classification__criterion' : ['entropy','gini']

}
classifier_rf = RandomForestClassifier(random_state=0, n_jobs=-1)
model = Pipeline([

        ('sampling', SMOTE()),

        ('classification', classifier_rf)

    ])
# Instantiate the grid search model

grid_search = GridSearchCV(estimator=model, param_grid=params, 

                          cv=4, n_jobs=-1, verbose=1)
%%time

grid_search.fit(X_train,y_train)
rf_best = grid_search.best_estimator_

rf_best
rf_best.fit(X_train, y_train)
rf_prob = rf_best.predict_proba(X_train)
cutoff_df = output_for_probability_range(y_train,rf_prob[:,1])

cutoff_df
prob_threshold_rf = get_optimal_threshold_curve(cutoff_df)

prob_threshold_rf
rf_y_train_pred_final = get_classification_report_tree(rf_best, y_train, X_train, prob_threshold_rf)

draw_roc(rf_y_train_pred_final.Target, rf_y_train_pred_final.Target_Prob)
y_val_pred = rf_best.predict(X_val)

recall_rf = precision_recall(y_val,y_val_pred)
y_preds = rf_best.predict(X_test)

precision_recall(y_test,y_preds)
get_confusion_matrix(y_test,y_preds)
weight_lr = 1/(100 - recall_lr*100)

weight_dt = 1/(100 - recall_dt*100)

weight_rf = 1/(100 - recall_rf*100)
def predict_probability(df) :

    df_lr = sm.add_constant(df.drop(['poutcome_unknown'],axis = 1))

    predict_prob_lr = np.array(res.predict(df_lr))

    predict_prob_dt = best_grid.predict_proba(df)[:,1]

    predict_prob_rf = rf_best.predict_proba(df)[:,1]

    return predict_prob_lr,predict_prob_dt,predict_prob_rf
def weighted_probability(lr_df,dt_df,rf_df) :

    size = len(lr_df)

    weighted_prob = []

    for i in range(0,size) :

        prob = weight_lr * lr_df[i] + weight_dt * dt_df[i] + weight_rf * rf_df[i]

        weighted_prob.append(prob)

    return weighted_prob
predict_prob_lr,predict_prob_dt,predict_prob_rf = predict_probability(X_train)
weighted_prob = weighted_probability(predict_prob_lr,predict_prob_dt,predict_prob_rf)
cutoff_df = output_for_probability_range(y_train,weighted_prob)

cutoff_df
prob_threshold_ens = get_optimal_threshold_curve(cutoff_df)

prob_threshold_ens
draw_roc(y_train, weighted_prob)
predict_prob_lr,predict_prob_dt,predict_prob_rf = predict_probability(X_test)

weighted_prob = weighted_probability(predict_prob_lr,predict_prob_dt,predict_prob_rf)
y_pred_final = pd.DataFrame({'Target' : y_test, 'Predicted_prob' : weighted_prob})

y_pred_final['Predicted'] = y_pred_final.Predicted_prob.apply(lambda x : 1 if x >= prob_threshold_ens else 0)

y_pred_final.head()
precision_recall(y_pred_final.Target,y_pred_final.Predicted)
get_confusion_matrix(y_pred_final.Target,y_pred_final.Predicted)