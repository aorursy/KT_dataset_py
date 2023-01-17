import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
%matplotlib inline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv('../input/weather-dataset/train.csv').rename(str.lower,axis='columns')
print(df.info())
print('-----------')
print(df.shape)
df.head(5)
sns.set(style="darkgrid")
palette=["#ff000d","#82cafc"]
ax = sns.countplot(y="label", data=df, palette=palette)
plt.title('Data balance')
df.label.value_counts()
total = len(df['label'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))
plt.show()
df6 = df.loc[:,('feature_6','label')]
df6.loc[:,'feature_6'] = df6.loc[:,'feature_6'].str.strip('a').astype('int64').sort_values()

fig = plt.figure(figsize=(15,5), constrained_layout=True)
gs = fig.add_gridspec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
sns.countplot(y=df6.loc[df.label == 0,'feature_6'], ax=ax1, data=df6,palette="Reds_d")
sns.countplot(y=df6.loc[df.label == 1,'feature_6'], ax=ax2,data=df6, palette="Blues_d")
ax2.set_xticklabels(range(0,800,100))
ax1.set_title('Sunny days')
ax2.set_title('Rainy days')

df_helper = df6.groupby('feature_6')['label'].value_counts(normalize=True)
df_helper = df_helper.mul(100).rename('percent').reset_index()
ax3 = sns.catplot(x='feature_6',y='percent',hue='label',kind='bar',
                data=df_helper,palette=palette,height=5,
                aspect=3)
ax3.ax.set_ylim(0,100)
for p in ax3.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    ax3.ax.text(txt_x,txt_y,txt)
plt.show()
df_helper = df.groupby('feature_9')['label'].value_counts(normalize=True)
df_helper = df_helper.mul(100).rename('percent').reset_index()
ax1 = sns.catplot(x='feature_9',y='percent',hue='label',kind='bar',
                data=df_helper,palette=palette,height=5,
                aspect=3)
ax1.ax.set_ylim(0,100)
ax1.ax.set_title('Percentage presentation of feature_9')
for p in ax1.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    ax1.ax.text(txt_x,txt_y,txt)
df_helper = df.groupby('year')['label'].value_counts(normalize=True)
df_helper = df_helper.mul(100).rename('percent').reset_index()
ax2 = sns.catplot(x='year',y='percent',hue='label',kind='bar',
                data=df_helper,palette=palette,height=5, aspect=3)
ax2.ax.set_ylim(0,100)
ax2.ax.set_title('Percentage presentation of year feature')
for p in ax2.ax.patches:
    txt = str(p.get_height().round(1)) + '%'
    txt_x = p.get_x() 
    txt_y = p.get_height()
    ax2.ax.text(txt_x,txt_y,txt)
plt.figure(figsize=(8,5))
f_13_c = df.feature_13.value_counts()
ax = sns.barplot(f_13_c.index, f_13_c.values, alpha=0.9,palette=palette)
ax.set_title('Frequencies of the categories in feature_13.')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
df_cat = df[['feature_5','feature_18','feature_19','label']].astype('category')

fig = plt.figure(figsize=(12,10))
plt.suptitle("Comparison between sunny days and rainy days with 'letters' features", ha='center',size= 18)
gs = fig.add_gridspec(3, 2)

cmap = plt.get_cmap('Reds')
color_reds = [cmap(i) for i in np.linspace(0, 1, 16)]

cmap = plt.get_cmap('Blues')
colors_blues = [cmap(i) for i in np.linspace(0, 1, 16)]

for i in range(0,3):
    labels = df_cat.loc[df_cat.label == 0].iloc[:,i].astype('category').cat.categories.tolist()
    counts = df_cat.loc[df_cat.label == 0].iloc[:,i].value_counts()
    sizes = [counts[j] for j in labels]
    plt.subplot(gs[i, 0])
    plt.pie(x=sizes, labels=labels, autopct='%1.1f%%',
        labeldistance=1.1,pctdistance=0.7 ,textprops={'fontsize': 8.5},shadow=True,radius=2, colors=color_reds)
    plt.title('%s Sunny day' %(df_cat.columns[i]),y=1.5)

for i in range(0,3):
    labels = df_cat.loc[df_cat.label == 1].iloc[:,i].astype('category').cat.categories.tolist()
    counts = df_cat.loc[df_cat.label == 1].iloc[:,i].value_counts()
    sizes = [counts[j] for j in labels]
    plt.subplot(gs[i, 1])
    plt.pie(x=sizes, labels=labels, autopct='%1.1f%%',
        labeldistance=1.1,pctdistance=0.7 ,textprops={'fontsize': 8.5},shadow=True,radius=2,colors=colors_blues)
    plt.title('%s Rainy day' %(df_cat.columns[i]),y=1.5)

fig.patch.set_edgecolor('black')  
fig.patch.set_linewidth('1') 
fig.tight_layout()    
fig.subplots_adjust(hspace = 1.2,left=0.05)
plt.show()
fig = plt.figure(figsize=(8, 5))
null_columns=df.columns[df.isnull().any()]
ax = df[null_columns].isnull().sum().plot(kind='bar',color='black')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()+2))
plt.title('Number of null values in the data')
plt.xlabel('Features with null values')
plt.ylabel('Number of null values')
plt.show()
continuous_vars = df.drop(['feature_5','feature_9','feature_13','feature_18','feature_19','year','label'],axis=1)
labels = df.label
continuous_vars['feature_6'] = continuous_vars['feature_6'].str.strip('a').astype('int64')
continuous_vars['feature_14'] = continuous_vars['feature_14'].str.strip('mm').astype('float64')
continuous_vars = (continuous_vars - continuous_vars.mean()) / (continuous_vars.std())
continuous_df = pd.concat([labels,continuous_vars],axis=1)
continuous_df = pd.melt(continuous_df,id_vars='label',var_name='features', value_name='value')
plt.figure(figsize=(20,10))
sns.boxplot(x='features', y='value', hue='label', data=continuous_df,palette="Set1")
plt.xticks(rotation=45)
plt.title('Box plots to reveal outliers')
plt.show()
fig3 = plt.figure(figsize=(25, 15))
gs1 = fig3.add_gridspec(3, 6)
cv = continuous_vars.drop(['feature_6'],axis=1)
cv_shape = np.array(cv.columns).reshape(3,6)
for i in range(3):
    for j in range(6):
        with sns.axes_style(style="darkgrid"):
            ax = fig3.add_subplot(gs1[i,j])
            ax.set_title(f"Distribution of {(cv_shape[i,j])}",fontsize=15)
            sns.distplot(cv[cv_shape[i,j]], color="darkblue" ,kde_kws={"color": "black"},axlabel=False)
plt.show()
skew_features = continuous_vars[['feature_0', 'feature_1', 'evaporation','feature_14', 'sunshine','windgustspeed', 'feature_21',
        'feature_23', 'feature_24']]

fig4 = plt.figure(figsize=(25, 15))
gs2 = fig4.add_gridspec(3, 3)
skew_features_shape = np.array(skew_features.columns).reshape(3,3)
old_settings = np.seterr(all='ignore')
for i in range(3):
    for j in range(3):
        with sns.axes_style(style="darkgrid"):
            ax = fig4.add_subplot(gs2[i,j])
            ax.set_title(f"Distribution after log-transform {(skew_features_shape[i,j])}",fontsize=15)
            sns.distplot(np.sqrt(skew_features[skew_features_shape[i,j]]), axlabel=False,color='firebrick',kde_kws={"color": "black"})
old_settings = np.seterr(all='warn')         
plt.show()
def train_preprocess(X):
    
    X['feature_14'] = X['feature_14'].str.strip('mm').astype('float64')
    X['feature_6'] = X['feature_6'].str.strip('a').astype('float64')
    c_df = ['float64','int64','float32','int32']
    letter_df = ['feature_5', 'feature_18', 'feature_19']
    remove = ['feature_1','feature_0','feature_24']
    for i in X.columns:
        if X[i].dtype == 'object':
            X[i] = X[i].astype('category')
        if i in letter_df:
            newl = []
            for j in range(ord('A'),ord('Q')):
                relative = 0
                relative = math.ceil(X[i].value_counts(normalize=True)[chr(j)] * 100)
                newl += chr(j) * relative
            for j in range(len(X[i])):
                if X[i].isnull()[j]:
                    X[i][j] = random.choice(newl)
        if X[i].dtype in c_df:
            X[i].fillna(X[i].median(), inplace=True)  
    ind = X[(X['feature_13'] == 'unknown')].index
    remove_index = []
    for i in X.columns:
        if i in remove:
            q1, q3, iqr = 0, 0, 0
            q1 = np.quantile(X[i], 0.25)
            q3 = np.quantile(X[i], 0.75)
            iqr = q3 - q1
            for j in range(len(X)):
                if X[i][j] < (q1 - 1.5 * iqr) or X[i][j] > (q3 + 1.5 * iqr):
                    remove_index.append(j)
    for i in range(len(ind)):
        if int(ind[i]) not in remove_index:
            remove_index.append(int(ind[i]))
    l = sorted(list(set(remove_index)))
    X = X.drop(X.index[l]).reset_index(drop=True)
    X['feature_13'] = X['feature_13'].astype('float64')
    return(X)
df = train_preprocess(df)
def k_fold_cv(x, y, clf, k = 10):
    data = pd.concat([x,y],axis=1)
    folds = np.array_split(data, k)
    new_list = []
    for i in range(k):
        train = folds.copy()
        validation = folds[i]
        del train[i]
        train = pd.concat(train, sort=False)
        x_train = train.values[:,:-1]
        x_validation = validation.values[:,:-1]
        y_train = train.values[:, -1]
        y_validation = validation.values[:, -1]
        clf.fit(x_train, y_train)
        y_proba = clf.predict_proba(x_validation)
        fpr, tpr, thresholds = roc_curve(y_validation, y_proba[:,1])
        new_list.append(auc(fpr, tpr))
    return np.mean(new_list)
y = df.label
X = df.drop(['label'],axis=1)
X_num = X.select_dtypes(exclude=['category'])
X_cat = X.select_dtypes(include=['category'])
columnsToEncode = X.select_dtypes(include=['category']).columns
mask = np.triu(np.ones_like(continuous_vars.corr(), dtype=np.bool))
f,ax_h = plt.subplots(figsize=(10, 10))
sns.heatmap(continuous_vars.corr(), mask=mask, annot=True, linewidths=.5, fmt= '.1f',ax=ax_h, cmap="Blues")
plt.show()
fig = plt.figure(figsize=(24,13),constrained_layout=True)
gs = fig.add_gridspec(4, 2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[3,0])
ax5 = fig.add_subplot(gs[0,1])
ax6 = fig.add_subplot(gs[1,1])
ax7 = fig.add_subplot(gs[2,1])
ax8 = fig.add_subplot(gs[3,1])

with sns.axes_style(style="darkgrid"):
    ax1.set_title('LinearReg between feature_0 and evaporation')
    sns.regplot(x='feature_0',y='evaporation', data=df, ax=ax1,line_kws={'color':'Red','alpha': 0.3})
    ax2.set_title('LinearReg between feature_1 and evaporation')
    sns.regplot(x='feature_1',y='evaporation', data=df, ax=ax2,line_kws={'color':'Red','alpha': 0.3})
    ax3.set_title('LinearReg between feature_12 and feature_11')
    sns.regplot(x='feature_12',y='feature_11', data=df, ax=ax3,line_kws={'color':'Red','alpha': 0.3})
    ax4.set_title('LinearReg between feature_17 and maxtemp')
    sns.regplot(x='feature_17',y='maxtemp', data=df, ax=ax4,line_kws={'color':'Red','alpha': 0.3})
    ax5.set_title('LinearReg between feature_16 and maxtemp')
    sns.regplot(x='feature_16',y='maxtemp', data=df, ax=ax5,line_kws={'color':'Red','alpha': 0.3})
    ax6.set_title('LinearReg between feature_17 and feature_8')
    sns.regplot(x='feature_17',y='feature_8', data=df, ax=ax6,line_kws={'color':'Red','alpha': 0.3})
    ax7.set_title('LinearReg between feature_1 and feature_0')
    sns.regplot(x='feature_1',y='feature_0', data=df, ax=ax7,line_kws={'color':'Red','alpha': 0.3})
    ax8.set_title('LinearReg between feature_17 and feature_16')
    sns.regplot(x='feature_17',y='feature_16', data=df, ax=ax8,line_kws={'color':'Red','alpha': 0.3})
plt.show()
v,std,c,n = [],[],[],[]
for i in X_num.columns:
    v.append(np.var(X_num[i]))
    std.append(np.std(X_num[i]))
    c.append(abs(y.corr(X_num[i])))
    n.append(str(i))
tempdf = pd.DataFrame(list(zip(n,v,std,c)),columns=['Features','Variance','STD','Correlation'])
tempdf = tempdf.sort_values(by=['STD'],ascending = False)

fig, ax1 = plt.subplots(figsize=(28, 13))
plt.title(label="Standard Deviation and Correlation vs Features",fontdict={'fontsize':20})
ax1.set_xlabel('Features',fontdict={'fontsize':20})
ax1.set_ylabel('<- Standard Deviation',fontdict={'fontsize':20})
ax1.plot(tempdf.values[:,0],
         tempdf.values[:,2],label ='Standard Deviation',color='Blue',marker='o')
ax1.tick_params(axis='y')
ax2 = ax1.twinx()
ax2.set_ylabel('<- Correlation with labels',fontdict={'fontsize':20})
ax2.plot(tempdf.values[:,0],
         tempdf.values[:,3],color='Red',marker='o')
ax2.tick_params(axis='y')
ax1.axhline(y=1,color='Orange')
ap = {'arrowstyle':'->', 'color': 'black', "connectionstyle":"arc3,rad=-0.2"}
bbx = {'pad':4, 'edgecolor':'orange', 'facecolor': 'orange', 'alpha':0.4}
ax1.annotate("Features with low variance", xy=('year', 1), xytext=('feature_9', 40),arrowprops=ap,bbox={'pad':4, 'edgecolor':'orange', 'facecolor': 'red', 'alpha':0.4})
ax2.annotate("Feature 1 correlation", xy=('feature_1', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_1']))), xytext=('feature_1', 0.2),arrowprops=ap,bbox=bbx)
ax2.annotate("Evaporation correlation", xy=('evaporation', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['evaporation']))), xytext=('feature_17', 0.1),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 12 correlation", xy=('feature_12', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_12']))), xytext=('feature_12', 0.3),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 17 correlation", xy=('feature_17', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_17']))), xytext=('feature_14', 0.05),arrowprops=ap,bbox=bbx)
ax2.annotate("Maxtemp correlation", xy=('maxtemp', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['maxtemp']))), xytext=('maxtemp', 0.15),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 0 correlation", xy=('feature_0', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_0']))), xytext=('year', 0.25),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 16 correlation", xy=('feature_16', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_16']))), xytext=('feature_17', 0.2),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 8 correlation", xy=('feature_8', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_8']))), xytext=('feature_8', 0.05),arrowprops=ap,bbox=bbx)
ax2.annotate("Feature 11 correlation", xy=('feature_11', float("%.6f" %(tempdf.set_index('Features', inplace=False).Correlation['feature_11']))), xytext=('feature_11', 0.35),arrowprops=ap,bbox=bbx)
fig.tight_layout()
plt.show()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_num, y)
X_scaled = pd.DataFrame(X_scaled,columns=X_num.columns,index=X_num.index)
select_feature = SelectKBest(chi2, k=5).fit(X_scaled, y)
newpd = pd.DataFrame(list(zip(select_feature.scores_,
                  X_scaled.columns)),columns=['score','features'])
newpd = newpd.sort_values(by=['score'],ascending=False)
newpd.reset_index(inplace=True)
newpd = newpd.drop(['index'],axis=1)
print(newpd)
reg = LogisticRegression(solver='liblinear')
X_num_t = X_num.copy()
score_list = []
diff = []
baseline = k_fold_cv(X_num_t,y,reg)
for i in newpd.features.iloc[21:0:-1]:
    auc1 = k_fold_cv(X_num_t,y,reg)
    score_list.append(auc1)
    diff.append(baseline - auc1)
    baseline = auc1
    X_num_t = X_num_t.drop(i,axis=1)

newl = list(newpd.features.iloc[21:0:-1].values)
plt.figure(figsize=(15, 6),constrained_layout=True)
plt.plot(newpd.features.iloc[21:0:-1], score_list)
plt.yticks(np.arange(0.8,0.88,step=0.02))
plt.ylabel('<- AUC')
plt.xlabel('Feature included before removing each')
plt.title('AUC as we drop least important feature and difference it makes')
for i in range(20):
    if diff[i] < 0.0009:
        if i % 2 == 0:
            plt.annotate('%0.5f' %diff[i], xy=(newl[i],score_list[i]), xytext=(newl[i],score_list[i] - 0.008),arrowprops=ap,bbox=bbx)
        else:
            plt.annotate('%0.5f' %diff[i], xy=(newl[i],score_list[i]), xytext=(newl[i],score_list[i] - 0.02),arrowprops=ap,bbox=bbx)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
X_lr = pd.concat([X_scaled,X_cat],axis=1)
X_lr = X_lr.drop(['feature_17','feature_24','feature_8','feature_1','maxtemp','feature_12'],axis=1)
X_lr = pd.get_dummies(X_lr, columns=columnsToEncode, drop_first=True)

reg_penalty = ['l1', 'l2']
powers = range(-10,0)
Cs = [10**p for p in powers]
reg_solver = ['liblinear']
grid = dict(penalty=reg_penalty, C=Cs, solver=reg_solver)
model = LogisticRegression()
grid = GridSearchCV(estimator=model, param_grid=grid, scoring='roc_auc', cv=10,n_jobs=-1)
grid_result = grid.fit(X_lr, y)

print("Best: %f using %s" % (grid_result.best_score_,
                             grid_result.best_params_))
fig = plt.figure(figsize=(8,6))
line1 = plt.plot(powers,grid_result.cv_results_['mean_test_score'][:10],label ='Lasso Regularization',color='blue', linewidth=2, alpha=0.6)
line2 = plt.plot(powers,grid_result.cv_results_['mean_test_score'][10:],label ='Ridge Regularization',color='green', linewidth=2, alpha=0.6)
plt.xlabel('Power values')
plt.ylabel('AUC ->')
plt.xticks(powers)
plt.title(label="Power & Penalty vs AUC")
plt.legend(loc=0)
plt.show()
X_gnb1 = X_num.copy()
X_gnb2 = X_num.drop(['feature_13','year','feature_9'],axis=1)
X_gnb3 = X_num.drop(['feature_13','year','feature_9','feature_1','feature_12','feature_17','maxtemp','feature_0'],axis=1)
X_gnb4 = X_num.drop(['feature_21','evaporation','feature_8','feature_13','year','feature_9','feature_1','feature_12','feature_17','maxtemp','feature_0'],axis=1)

gnb_models = [X_gnb1,X_gnb2,X_gnb3,X_gnb4]
names = ['Numerical features','Continuous features','Continuous features\n after removing\n correlated features\n > 0.8 corr','Continuous features\n after removing\n correlated features\n > 0.6 corr']
results = []
model = GaussianNB()
for i in gnb_models:
    cv_results = cross_val_score(model,i, y, cv=10, scoring='roc_auc')
    results.append(cv_results)
fig = plt.figure(figsize=(10,7))
sns.axes_style(style="darkgrid") 
ax = sns.boxplot(data=results,orient='v').set(xticklabels=names)
plt.title('Gaussian Naive Bayes models comparison',fontsize=15)
fig.text(0.02, 0.5, 'AUC ->', va='center', rotation='vertical',fontsize=10)
plt.show()
X_svm = scaler.fit_transform(X_gnb3, y)
X_svm = pd.DataFrame(X_svm,columns=X_gnb3.columns,index=X_gnb3.index)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=10,n_jobs=-1)
grid_result = grid.fit(X_svm, y)

print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
fig = plt.figure(figsize=(8,6))
line1 = plt.plot(c_values,grid_result.cv_results_['mean_test_score'][0:21:2],label ='linear kernel',color='blue', linewidth=2, alpha=0.6)
line2 = plt.plot(c_values,grid_result.cv_results_['mean_test_score'][1:20:2],label ='poly kernel',color='green', linewidth=2, alpha=0.6)
plt.xlabel('C values')
plt.ylabel('AUC ->')
plt.xticks(c_values)
plt.title(label="C values & kernel vs AUC")
plt.legend(loc=0)
plt.show()
X_num = X_num.drop(['feature_13'],axis=1)
ne = list(range(200,1400,200))
res = []
for i in ne:
    rfc = RandomForestClassifier(n_estimators=i, criterion='entropy', max_depth=100,max_features='sqrt',bootstrap=True)
    res.append(cross_val_score(rfc, X_num, y, cv=10, scoring='roc_auc',n_jobs=-1).mean())
print('Best AUC: %.3f with 1200 numbers of trees' %(max(res)))
plt.style.use('seaborn-darkgrid')
plt.plot(ne,res,color='red', linewidth=2, alpha=0.6)
plt.xticks(ne)
plt.xlabel('Number of trees in the forest')
plt.ylabel('AUC')
plt.title(label="Number of trees in the forest vs AUC")
plt.show()
def KfoldPlot(X, y, clf, name):
    
    X = X.values
    y = y.values
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(10, 4),constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    for i, (train, validation) in enumerate(kf.split(X, y)):
        clf.fit(X[train], y[train])
        y_proba = clf.predict_proba(X[validation])
        fpr, tpr, thresholds = roc_curve(y[validation], y_proba[:,1])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
        ax1.plot(fpr, tpr, color='#D3D3D3', alpha=.8)
    X_train_cm, X_validation_cm, y_train_cm, y_validation_cm = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train_cm, y_train_cm)
    cm = confusion_matrix(y_validation_cm, clf.predict(X_validation_cm))
    tn, fp, fn, tp = cm.ravel()
    cm = np.array([[tn,fn],[fp,tp]])
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    ax2 = sns.heatmap(cm,annot=True,fmt='g',cmap='Blues')
    ax2.set(title="Confusion Matrix")
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=.8, color='darkBlue')
    ax1.plot(mean_fpr, mean_tpr, color='Red', label=r'Mean ROC (AUC = %0.3f)' % (mean_auc), lw=2, alpha=.9)
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver Operating Characteristic")
    ax1.legend(loc="lower right")
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth('1')  
    ax1.patch.set_facecolor('white')
    ax1.grid(False)
    print("Validation AUC = %0.3f for %s model" %(mean_auc, name))
    plt.show()

reg = LogisticRegression(C=0.1,solver='liblinear',penalty='l1')
KfoldPlot(X_lr, y, reg, 'Logistic Regression')
gnb = GaussianNB()
KfoldPlot(X_gnb3, y, gnb, 'Gaussian Naive Bayes')
svm = SVC(C=0.3, kernel='linear',probability=True)
KfoldPlot(X_svm, y, svm, 'Support Vector Machine')
rfc = RandomForestClassifier(n_estimators=1200, criterion='entropy', max_depth=100,max_features='sqrt',bootstrap=True)
KfoldPlot(X_num, y, rfc, 'Random Forest Classifier')
def test_preprocess(X):
    X['feature_14'] = X['feature_14'].str.strip('mm').astype('float64')
    X['feature_6'] = X['feature_6'].str.strip('a').astype('float64')
    c_df = ['float64','int64','float32','int32']
    letter_df = ['feature_5', 'feature_18', 'feature_19']
    for i in X.columns:
        if X[i].dtype == 'object':
            X[i] = X[i].astype('category')
        if i in letter_df:
            newl = []
            for j in range(ord('A'),ord('Q')):
                relative = 0
                relative = math.ceil(X[i].value_counts(normalize=True)[chr(j)] * 100)
                newl += chr(j) * relative
            for j in range(len(X[i])):
                if X[i].isnull()[j] == True:
                    X[i][j] = random.choice(newl)
        if (X[i].dtype in c_df):
            X[i].fillna(X[i].median(), inplace=True)  
    return(X)
X_test = pd.read_csv('../input/test-without-target/test_without_target.csv').rename(str.lower,axis='columns')
X_test = X_test.drop(X_test.columns[0], axis=1)
X_test = test_preprocess(X_test)
X_test_num = X_test.select_dtypes(exclude=['category'])
rfc.fit(X_num,y)
pred_proba = rfc.predict_proba(X_test_num)[:,1]
#pd.DataFrame(pred_proba).to_csv("Submission.csv")