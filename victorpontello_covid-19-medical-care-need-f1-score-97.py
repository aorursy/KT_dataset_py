import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_excel('../input/covid19/dataset.xlsx')
df.head()
df.describe()
# All the features
[col for col in df.columns]
all_null = [feature for feature in df.columns if df[feature].isnull().sum() == len(df)] 
# all these features are completelly null, so they are not important for the analysis
all_null.append('Patient ID')
print(all_null)
df.drop(all_null, inplace=True, axis=1)
# all the not numerical features
[feature for feature in df.columns if not np.issubdtype(df[feature].dtype, np.number) if feature not in 'SARS-Cov-2 exam result']
# saves the non numeric features
not_numeric = [feature for feature in df.columns if not np.issubdtype(df[feature].dtype, np.number) if feature not in 'SARS-Cov-2 exam result']
# transforms some possible string features into numeric, if the string is in numerical form
df.update(df.apply(pd.to_numeric, errors='coerce'))
# shows the unique values of all the features, which are not numbers
df_subset = df.select_dtypes(exclude=[np.number]).copy()
[print(f'{col}: {df_subset[col].value_counts().index}') for col in df_subset.columns]
# turn nao realizado into NaN
df['Urine - pH'][df['Urine - pH'] == 'Não Realizado'] = np.nan
df['Urine - pH'].value_counts()
# transforms the string into zero
df['Urine - Leukocytes'][df['Urine - Leukocytes'] == '<1000'] = 0
df['Urine - Leukocytes'].value_counts()
 # defines the largest Leukocytes values to be 300.000
df['Urine - Leukocytes'][df['Urine - Leukocytes'] > 300000] = 300000
sns.distplot(df['Urine - Leukocytes'][df['Urine - Leukocytes'] < 500000].apply(int))
# separates the Leukocytes into categories accordingly to the distribution of it
df['Urine - Leukocytes'] = pd.qcut(df['Urine - Leukocytes'], q = [0.1, 0.2, 0.3, 0.4, 0.75, 1])
# creating dummy
#not_numeric = [feature for feature in df.columns if not np.issubdtype(df[feature].dtype, np.number) if feature not in 'SARS-Cov-2 exam result']
is_string = [col for col in df_subset.columns if col not in 'SARS-Cov-2 exam result']
for feature in is_string:
    df[[(str(feature) + '_' + str(col)) for col in pd.get_dummies(df[feature]).columns]] = pd.get_dummies(df[feature])
    df.drop(feature,axis=1,inplace=True)
df.head()
# shows the feature title and it's type
[print(f'{col} {type(col)}') for col in df.columns]
# defines the layout to the plotting
sns.set_context('paper', font_scale=1.5)
sns.set_style('whitegrid')
df['Patient age quantile'].max()
df['Patient age quantile'].min()
# plot about the age distribution for someone with Cov-2 and without it
fig = plt.figure(figsize=(10,7))
fig = sns.distplot(df['Patient age quantile'][df['SARS-Cov-2 exam result']=='negative'], kde=True, hist=False, color='blue', label='negative')
fig = sns.distplot(df['Patient age quantile'][df['SARS-Cov-2 exam result']=='positive'], kde=True, hist=False, color='red', label='positive')
fig = plt.xticks(np.arange(0, 20, step=1))
fig = plt.xlim(df['Patient age quantile'].min(),df['Patient age quantile'].max())
fig = plt.legend(loc='best')


fig = plt.figure(figsize=(12,7))
fig = sns.countplot(x='Patient age quantile',data=df[df['Patient age quantile']<=4], palette='viridis',
             hue='SARS-Cov-2 exam result')
for p in fig.patches:
    fig.annotate('{:.0f}'.format(round(p.get_height())), (p.get_x()+0.1, p.get_height()+0.2), fontsize=20)
fig = plt.ylim(0,420)
# gets the numerical features
num_features = [feature for feature in df.columns if np.issubdtype(df[feature].dtype, np.number)]
print(f'length numerical features: {len(num_features)}')
print(f'length total features: {len(df.columns)}')
def plot_pie(data, feature):
    plt.pie(data[feature].value_counts(), autopct='%1.1f%%',startangle=90, colors=['#aaffd5', '#fec0cb'],
        labels=data[feature].value_counts().index)
    plt.legend()
    print(df[feature].value_counts())
plot_pie(df,'Patient addmited to intensive care unit (1=yes, 0=no)' )
plot_pie(df,'Patient addmited to semi-intensive unit (1=yes, 0=no)' )
plot_pie(df,'Patient addmited to regular ward (1=yes, 0=no)')
patient_cols = [col for col in df.columns if 'Patient addmited' in col]
df['Medical_care_needed'] = (df[patient_cols]==1).any(axis=1).astype(int)
df['Medical_care_needed'].sum()
plot_pie(df, 'Medical_care_needed')
fig = plt.figure(figsize=(5,8))
fig = sns.countplot(x='Medical_care_needed',data=df[df['SARS-Cov-2 exam result']=='positive'], palette='viridis')
for p in fig.patches:
    fig.annotate('{:.1f}%'.format(round(p.get_height()/len(df[df['SARS-Cov-2 exam result']=='positive'])*100)), (p.get_x()+0.2, p.get_height()+0.2), fontsize=20)
#plt.ylim(0,420)
# feature engineering about the people, which are going to need semi-intensive and intensive care
df['Medical_semi_int'] = (df[patient_cols[1:]]==1).any(axis=1).astype(int)
df['Medical_semi_int'].sum()
plot_pie(df, 'Medical_semi_int')
# plot about the missing values from the dataset
fig, ax = plt.subplots(nrows=4, ncols=1)
fig.set_figheight(40)
fig.set_figwidth(15)
for i in range(len(patient_cols)):
    ax[i].set_title(patient_cols[i])
    sns.heatmap(data=df[df[patient_cols[i]]==1].isna(), cmap='viridis', annot=False, cbar=False, ax=ax[i], 
               xticklabels=True, yticklabels=False)
ax[i+1].set_title('Whole dataset')
sns.heatmap(data=df.isna(), cmap='viridis', annot=False, cbar=False, ax=ax[i+1], 
               xticklabels=True, yticklabels=False)
def get_corr(data,features,base_feature):
    '''
    gets the correlation between the two input features
    features => features from which the correlation is going to be calculated
    base_feature => the base feature to be correlated
    data => the data from there the calculations are going to be done
    '''
    data_to = pd.Series()
    for feature in features:
        try:
            data_to[feature] = data[[base_feature,feature]].dropna().corr().iloc[1,0]
        except:
            continue
    return data_to.dropna().sort_values(ascending=False)
    
med_care = get_corr(df,num_features,'Medical_care_needed')
med_reg = get_corr(df,num_features,patient_cols[0])
med_semi = get_corr(df,num_features,patient_cols[1])
med_int = get_corr(df,num_features,patient_cols[2])
def get_tops(top, med_reg, med_semi, med_int):
    '''
    function gets the top correlated features from regular, semi-intensive and intensive care
    
    med_reg => correlations to medical care needed
    med_semi => correlations to semi-intensive care needed
    med_semi => correlations to intensive care needed
    '''
    top_reg_semi_int = [feature for feature in med_care.index if feature in med_reg.index[:top] and feature in med_semi.index[:top] and feature in med_int.index[:top]]
    top_semi_int = [feature for feature in med_care.index if feature in med_semi.index[:top] and feature in med_int.index[:top]]
    return top_reg_semi_int, top_semi_int
top10_reg_semi_int, top10_semi_int = get_tops(10, med_reg, med_semi, med_int)
top30_reg_semi_int, top30_semi_int = get_tops(30, med_reg, med_semi, med_int)
top50_reg_semi_int, top50_semi_int = get_tops(50, med_reg, med_semi, med_int)
top70_reg_semi_int, top70_semi_int = get_tops(70, med_reg, med_semi, med_int)
def print_tops(top_reg_semi_int):
    '''
    shows the top correlations
    
    top_reg_semi_int => calculated most important correlations
    '''
    for feature in top_reg_semi_int:
        print(feature)
        print(f'regular: {med_reg[feature]}')
        print(f'semi: {med_semi[feature]}')
        print(f'intensive: {med_int[feature]}\n')
    print('#########################\n')
print_tops(top10_reg_semi_int)
print_tops(top30_reg_semi_int)
print_tops(top50_reg_semi_int)
print_tops(top70_reg_semi_int)
# check tencences
def get_tend_tops(top_reg_semi_int):
    '''
    gets the risk tendence of each feature
    
    top_reg_semi_int => calculated most important correlations
    '''
    tendences = pd.Series()
    for feature in top_reg_semi_int:
        if med_reg[feature] < med_semi[feature] and med_semi[feature] < med_int[feature]:
            tendences[feature] = med_reg[feature], med_semi[feature] ,med_int[feature]
        elif med_reg[feature] > med_semi[feature] and med_semi[feature] > med_int[feature]:
            tendences[feature] = med_reg[feature], med_semi[feature] ,med_int[feature]
    return tendences
# shows the tendences
tendences = get_tend_tops(top50_reg_semi_int)
tendences2 = get_tend_tops(top70_reg_semi_int)
print(tendences)
print(f'\nnumber of features: {len(tendences.index)}')
print('\n')
print(tendences2)
print(f'\nnumber of features: {len(tendences2.index)}')
print(df[tendences.index].isna().sum().sort_values(ascending=False))
print(f'\nlength of df_pos dataset: {len(df)}\n')
print( 1 - df[tendences.index].isna().sum().sort_values(ascending=False) / len(df))
print(df[tendences2.index].isna().sum().sort_values(ascending=False))
print(f'\nlength of df_pos dataset: {len(df)}\n')
print( 1 - df[tendences2.index].isna().sum().sort_values(ascending=False) / len(df))
fig = plt.figure(figsize=(20,3))
sns.heatmap(df[list(tendences.index)+patient_cols].corr().iloc[-3:,:-3], annot=True, cmap='viridis', cbar=False)
fig = plt.figure(figsize=(25,3))
sns.heatmap(df[list(tendences2.index)+patient_cols].corr().iloc[-3:,:-3], annot=True, cmap='viridis', cbar=False)
list(tendences.index)+patient_cols
list(tendences2.index)+patient_cols
# getting the best features from the dataset - 2 
usable_features2 = [feature for feature in list(tendences2.index) if feature not in patient_cols]
usable_features2
len(df[usable_features2].dropna())
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler2 = StandardScaler()
# splitting the data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import confusion_matrix,classification_report
import lightgbm as lgb
# creating the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import regularizers
# imputing median values on the missing values
df_imp = df.copy()
neg = df_imp[df_imp['SARS-Cov-2 exam result']=='negative']
neg.fillna(neg.median(), inplace=True)
pos = df_imp[df_imp['SARS-Cov-2 exam result']=='positive']
pos.fillna(pos.median(), inplace= True)

df_median = pd.concat([neg,pos])
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn import linear_model
def get_nan_features(data):
    nan = data.isna().sum().sort_values(ascending=False)
    nan_data = [(nan.index[i], nan.values[i]) for i in range(len(nan)) if nan[i] > 0]
    nan = df_median.isna().sum().sort_values(ascending=False)
    l = []
    if nan[0] > 0:
        print('these features still have NaN values:')
        for feature, val in nan_data:
            print(f'{feature}: {val}')
            l.append(feature)
            
    else:
        print('no nan values')
    return l
df_median.fillna(df_median[get_nan_features(df_median)].median(), inplace= True)
get_nan_features(df_median)
# checking the existence of missing values
df_median.isna().sum().sort_values(ascending=False)[:10]
drop_features = [feature for feature in df_median.columns if 'Medical' in feature or 'Patient addmi' in feature or 'SARS-Cov-2 exam result' in feature]
print('Features, which are going to be droped')
drop_features
X = df_median.drop(drop_features, axis=1)
X['SARS-Cov-2 exam result'] = pd.get_dummies(df_median['SARS-Cov-2 exam result'],drop_first=True)
y = df_median['Medical_semi_int']
X, y = SMOTE().fit_resample(X, y)
for feature in X.columns:
    print(feature)
print('Features, which are going to be droped for the training')
to_drop = [feature for feature in df_median.columns if 'detected' in feature or 'positive' in feature or 'negative' in feature]
to_drop
X.drop(to_drop, axis=1, inplace=True)
# All the features which are going to be used to train
features = [col for col in X.columns]
print(len(features))
features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train = scaler2.fit_transform(X_train)
X_test = scaler2.transform(X_test)
print(f'features X_train: {len(X_train[1])}\nfeatures X_test: {len(X_test[1])}')
plot_pie(pd.DataFrame(y), pd.DataFrame(y).columns[0])
adac = AdaBoostClassifier()
adac.fit(X_train,y_train)
adac_pred = adac.predict(X_test)
print(classification_report(y_test, adac_pred))
lgbc = lgb.LGBMClassifier()
lgbc.fit(X_train, y_train,)
lgbc_pred=lgbc.predict(X_test)
print(classification_report(y_test, np.round(lgbc_pred)))
from xgboost import XGBClassifier, plot_importance
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
xgbc_pred=xgbc.predict(X_test)
print(classification_report(y_test, np.round(xgbc_pred)))
def lr_scheduler(epoch, lr):
    decay_rate = 0.999
    decay_step = 10
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr
import tensorflow.keras.backend as K

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def build_model(hp):
    '''
    function that creates the model for the random search
    
    input:
    hp - objetc from the HyperParameter class from the kerastuner library
    
    output:
    model - created model with the random hyperparameters for the random search'''
    
    model = Sequential()

    model.add(Dense(hp.Int('units_1',min_value=30, max_value=100, sampling='linear', default=75),
                    activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
    model.add(Dropout(rate=hp.Choice('drop_out_1',values=[0.,0.1,0.2,0.3,0.4,0.5], default=0.3)))

    model.add(BatchNormalization())
    model.add(Dense(hp.Int('units_2',min_value=30, max_value=100, sampling='linear', default=75),
                    activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
    model.add(Dropout(rate=hp.Choice('drop_out_2',values=[0.,0.1,0.2,0.3,0.4,0.5], default=0.3)))

    model.add(BatchNormalization())
    model.add(Dense(hp.Int('units_3',min_value=20, max_value=50, sampling='linear', default=50),
                    activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
    model.add(Dropout(rate=hp.Choice('drop_out_3',values=[0.,0.1,0.2,0.3], default=0.3)))

    model.add(BatchNormalization())
    model.add(Dense(hp.Int('units_4',min_value=5, max_value=20, sampling='linear', default=20),
                    activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
    model.add(Dropout(rate=hp.Choice('drop_out_4',values=[0.,0.1,0.2,0.3], default=0.3)))

    model.add(BatchNormalization())
    model.add(Dense(1, activation=hp.Choice('last_activation',['sigmoid','hard_sigmoid'])))

    adam = optimizers.Adam(learning_rate=hp.Float( 'learning_rate',
                                                    min_value=1e-6,
                                                    max_value=1e-1,
                                                    sampling='LOG',
                                                    default=1e-3), 
                           beta_1=0.9, beta_2=0.999, amsgrad=True)
              
    model.compile(optimizer=adam, loss=hp.Choice('loss_function', ['binary_crossentropy','hinge','squared_hinge']), metrics=[get_f1])
    return model
# definition of the early stop parameters
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
class_weight = {0: 1., 1: 1.}
model = Sequential()

model.add(Dense(90,activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
model.add(Dropout(0.3))

model.add(BatchNormalization())
model.add(Dense(70,activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
model.add(Dropout(0.3))

model.add(BatchNormalization())
model.add(Dense(25,activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(15,activation='relu', kernel_regularizer=regularizers.l2(l=0.1)))
model.add(Dropout(0.1))

model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(learning_rate=3e-5, 
                       beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[get_f1])
model.fit(x=np.array(X_train), y=np.array(y_train), validation_data=(np.array(X_test), np.array(y_test)),
             batch_size = 128, epochs = 500)#, callbacks=[LearningRateScheduler(lr_scheduler, verbose=1)])
model_0 = model
# model_1 = tuner.get_best_models(num_models=5)[0]
# model_2 = tuner.get_best_models(num_models=5)[1]
# model_3 = tuner.get_best_models(num_models=5)[2]
# model_4 = tuner.get_best_models(num_models=5)[3]
# model_5 = tuner.get_best_models(num_models=5)[4]
models = [model_0]#, model_1, model_2, model_3, model_4, model_5]
print('#############################################################################')
print('model_0')
print('#############################################################################')
predictions = model_0.predict_classes(X_test)
print(classification_report(y_test, predictions))
print('#############################################################################')
# print('model_1')
# print('#############################################################################')
# predictions = model_1.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('model_2')
# print('#############################################################################')
# predictions = model_2.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('model_3')
# print('#############################################################################')
# predictions = model_3.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('model_4')
# print('#############################################################################')
# predictions = model_4.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('model_5')
# print('#############################################################################')
# predictions = model_5.predict_classes(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('AdaBoost')
# print('#############################################################################')
# predictions = adac.predict(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')
# print('LightGBM')
# print('#############################################################################')
# predictions = lgbc.predict(X_test)
# print(classification_report(y_test, predictions))
# print('#############################################################################')

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_score,average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
test_data = {}
for i in range(len(models)):
    test_data['model_'+str(i)] = np.array(models[i].predict_classes(X_test))
test_data['AdaBoost'] = np.array(adac.predict(X_test))
test_data['LightGBM'] = np.array(lgbc.predict(X_test))
test_data['XGBoost'] = np.array(xgbc.predict(X_test))
test_data_prob = {}
for i in range(len(models)):
    test_data_prob['model_'+str(i)] = np.array(models[i].predict_proba(X_test))
test_data_prob['AdaBoost'] = np.array(adac.predict_proba(X_test)[:,1])
test_data_prob['LightGBM'] = np.array(lgbc.predict_proba(X_test)[:,1])
test_data_prob['XGBoost'] = np.array(xgbc.predict_proba(X_test)[:,1])
def get_scores(y_test, y_pred):
    cache = {}
    cache['accuracy'] = accuracy_score(y_test, y_pred)
    cache['precision'] = precision_score(y_test, y_pred)
    cache['recall'] = recall_score(y_test, y_pred)
    cache['roc'] = roc_auc_score(y_test, y_pred)
    cache['f1'] = f1_score(y_test, y_pred)
    return cache
def performances(models, y_test):
    perf_data={}
    for model,y_pred in models.items():
        perf_data[model] = get_scores(y_test,y_pred)
    return pd.DataFrame(perf_data)
def plot_roc_curve(dict_pred,y_test):
    f, ax = plt.subplots(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='silver')
    plt.title('ROC Curve', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    i=0
    for model, pred in dict_pred.items():
        i+=1
        roc_score = roc_auc_score(y_test, pred)
        fpr, tpr, thresholds = roc_curve(y_test, pred) 
        sns.lineplot(x=fpr, y=tpr, color=sns.color_palette("magma", 10)[-i], 
                     linewidth=2, label= f"ROC-AUC {model}= {round(roc_score*100,2)}%", ax=ax)
def plot_pr_curve(dict_pred,y_test):
    f, ax = plt.subplots(figsize=(8, 8))
    plt.title('PR-Curve', fontsize=20)
    plt.xlabel('Precision', fontsize=20)
    plt.ylabel('Recall', fontsize=20)
    i=0
    for model, pred in dict_pred.items():
        i+=1
        average_precision = average_precision_score(y_test, pred)
        fpr, tpr, thresholds = precision_recall_curve(y_test, pred)
        sns.lineplot(x=fpr, y=tpr, color=sns.color_palette("magma", 10)[-i],
                     linewidth=2, label= f"PR-AUC {model}= {round(average_precision*100,2)}%", ax=ax)
perf_data = performances(test_data,y_test)
plt.figure(figsize=(15,10))
sns.heatmap(perf_data*100, annot=True, cmap='coolwarm',fmt='.3g')
plot_roc_curve(test_data_prob,y_test)
plot_pr_curve(test_data_prob,y_test)
n = 50
cache_test = {'model_'+str(i):[] for i in range(len(models))}
cache_test['AdaBoost'] = []
cache_test['LightGBM'] = []
cache_test['XGBoost'] = []

for j in range(n):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.30, random_state=j)
    X_train_ = scaler2.transform(X_train_)
    X_test_ = scaler2.transform(X_test_)
    
    # getting the performance of the 5 best models from the random search
    for i in range(len(models)):
        predictions = models[i].predict_classes(X_test_)
        cache_test['model_'+str(i)].append(float(classification_report(y_test_, predictions)[148:153]))
        
    adac_pred = adac.predict(X_test_)
    cache_test['AdaBoost'].append(float(classification_report(y_test_, adac_pred)[148:153]))
    lgbc_pred = lgbc.predict(X_test_)
    cache_test['LightGBM'].append(float(classification_report(y_test_, lgbc_pred)[148:153]))
    xgbc_pred = xgbc.predict(X_test_)
    cache_test['XGBoost'].append(float(classification_report(y_test_, xgbc_pred)[148:153]))
       
# transforming the generated dictionaries into pandas dataframes
cache_test = pd.DataFrame(cache_test)
fig, ax = plt.subplots(nrows=2, ncols=1)
fig.set_figheight(20)
fig.set_figwidth(15)
best_test = f'Best f1-Score from {np.argmax(cache_test.mean())}: {round(cache_test.mean().max()*100,2)}%'
ax[0].set_title('Cross Validation along with multiple test_set Samples\n' + best_test)
sns.lineplot(data=cache_test, ax=ax[0], dashes=False)
ax[1].set_title(f'Mean F1-Score values along {n} sample tests\n' + best_test)
ax[1].bar(x=cache_test.mean().index, height=cache_test.mean().values)

for i in range(len(ax)):
    if i%2==1:
        for p in ax[i].patches:
            ax[i].annotate(f'{round(p.get_height()*100,2)}%', (p.get_x()+0.3, p.get_height()+0.02), fontsize=15)
n=50
performance_values = []
for j in range(n):
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.30, random_state=j)
    X_train_ = scaler2.transform(X_train_)
    X_test_ = scaler2.transform(X_test_)
    test_data = {}
    for i in range(len(models)):
        test_data['model_'+str(i)] = np.array(models[i].predict_classes(X_test_))
    test_data['AdaBoost'] = np.array(adac.predict(X_test_))
    test_data['LightGBM'] = np.array(lgbc.predict(X_test_))
    test_data['XGBoost'] = np.array(xgbc.predict(X_test_))
    performance_values.append(performances(test_data,y_test_))
perf_data = pd.concat(performance_values)
perf_data.groupby(perf_data.index).mean()
plt.figure(figsize=(15,10))
sns.heatmap(perf_data.groupby(perf_data.index).mean()*100, annot=True, cmap='coolwarm',fmt='.4g')
abstract = np.mean(perf_data.groupby(perf_data.index).mean(),axis=0).sort_values(ascending=False)*100
print(abstract)
model_0.save("model_MedCare_random_search_v8.h5")
lgbc.booster_.save_model("LightGBM_model.txt")
xgbc.save_model('XGBoost_model.model')
