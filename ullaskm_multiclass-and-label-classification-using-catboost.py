import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

from imblearn.over_sampling import SMOTE,SMOTENC,SVMSMOTE

from imblearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score,make_scorer

from sklearn.metrics import precision_score, recall_score, confusion_matrix,classification_report

from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from datetime import date

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/train.csv')

df_test = pd.read_csv('../input/hackerearth-ml-challenge-pet-adoption/test.csv')

df_train.head()
profile = ProfileReport(df_train,title='Detailed Customer Report')

profile.to_widgets()
def evaluate():

    # f1 score

    s1 = f1_score(y_test,y_pred_breed,average='weighted')

    s2 = f1_score(y_test,y_pred_pet,average='weighted')

    score = 100*((s1+s2)/2)

    return score



def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



def generate_model_report(y_actual, y_predicted):

    print("Accuracy = " , accuracy_score(y_actual, y_predicted))

    print("Precision = " ,precision_score(y_actual, y_predicted,average='weighted'))

    print("Recall = " ,recall_score(y_actual, y_predicted,average='weighted'))

    print("F1 Score = " ,f1_score(y_actual, y_predicted,average='weighted'))

    pass



def generate_auc_roc_curve(y_test, y_score,n_classes):

   # Compute ROC curve and ROC area for each class

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])



    # Plot of a ROC curve for a specific class

    for i in range(n_classes):

        plt.figure()

        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('Receiver operating characteristic example')

    pass
# 1) To clean the data and see which are the redundant or unnecessary cols

def del_col(col,data):

    clean_data = data.drop(col, axis=1)

    return clean_data



# 2) Dropping the duplicates from the dataset.

def del_duplicates(data):

    clean_data = data.drop_duplicates(keep='first')

    return clean_data



# 3) Imputing missing data

def impute_col(data,filler):

    data.fillna(filler,inplace=True)

    return data



# 4) Typecasting Variables

def typecast_col(col,data,types):

    clean_data = data.col.astype(types)

    return clean_data

  

# 5) To Replace the spaces between the strings with '_' and also converting all strings to LowerCase

def convert_case(col,data,chars):

    data = data.str.replace(' ',chars) 

    data = data.str.lower() 

    return data



# 6) Encoding using Label Encoder or OHE which converts categorical features to numerical features

def label_encoder(data):

    le = LabelEncoder()

    data = le.fit_transform(data)

    return data

# Removing Unnecessary Columns

X = del_col('pet_id',df_train)



# Generating new feature 

X[['issue_date','listing_date']] = X[['issue_date','listing_date']].apply(pd.to_datetime) #if conversion required

X['diff_days'] = (X['listing_date'] - X['issue_date']).dt.days

X = del_col('issue_date',X)

X = del_col('listing_date',X)



# Imputing missing values with new category 

X['condition'] = impute_col(X['condition'],3.0)

X['condition'] = X['condition'].astype('int')



# Standardization - converting cm to mts

# X['height(cm)'] = X['height(cm)']*0.01

X['length(cm)'] = X['length(m)'].apply(lambda x: x*100)

X = del_col('length(m)',X)

# replace all 0 length with mean of lengths

val = X['length(cm)'].mean()

X['length(cm)'] = X['length(cm)'].replace(to_replace=0, value=val)

quantile_list = [0, .25, .5, .75, 1.]

quantiles = X['length(cm)'].quantile(quantile_list)

quantiles
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

X['length_label'] = pd.qcut(X['length(cm)'],q=quantile_list, labels=quantile_labels)

X.head()
X['diff_days'] = abs(X['diff_days'])

X['diff_days'] =np.array(np.array(X['diff_days']) / 365.)
# Encoding category using One Hot Encoding

X = pd.concat([X,pd.get_dummies(X['condition'], prefix='condition')],axis=1)

X = pd.concat([X,pd.get_dummies(X['X2'], prefix='X2')],axis=1)

X = pd.concat([X,pd.get_dummies(X['X1'], prefix='X1')],axis=1)

X = pd.concat([X,pd.get_dummies(X['color_type'], prefix='color_type')],axis=1)

X = pd.concat([X,pd.get_dummies(X['length_label'], prefix='length_label')],axis=1)



X = del_col('condition',X)

X = del_col('color_type',X)

X = del_col('X2',X)

X = del_col('X1',X)

X = del_col('length(cm)',X)

X = del_col('length_label',X)

X['breed_category'] = X['breed_category'].astype('int')

X['pet_category'] = X['pet_category'].astype('int')

X.head()
Y1 = X['breed_category']

Y2 = X['pet_category']



#Splitting up for MultiLabel Classification

X1 = X.drop(['pet_category','breed_category'], axis=1)

X2 = X.drop(['pet_category','breed_category'], axis=1)
X1.head()
X2.head()
smote = SMOTE('auto',random_state=42)

X_train1_smote,y_train1_smote = smote.fit_resample(X1,Y1)
smote1 = SMOTE('auto',random_state=42)

X_train2_smote,y_train2_smote = smote1.fit_resample(X2,Y2)
from collections import Counter

print("Before SMOTE :", Counter(Y1))

print("Before SMOTE :", Counter(y_train1_smote))
from collections import Counter

print("Before SMOTE :", Counter(Y2))

print("Before SMOTE :", Counter(y_train2_smote))
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold

categorical_features_indices=[0]

rf1_2 = CatBoostClassifier(learning_rate=0.055, 

                          n_estimators=1000, 

                          subsample=0.075, 

                          max_depth=3, 

                          verbose=100,

                          l2_leaf_reg = 7,

                          bootstrap_type="Bernoulli",

                          class_weights=[1, 1, 1],

                          loss_function='MultiClass')

#                           eval_metric='F1')



kf = StratifiedKFold(n_splits=7,shuffle=True,random_state=99)

f1 = []



for fold,(t_id,v_id) in enumerate(kf.split(X_train1_smote,y_train1_smote)):

    tx = X_train1_smote.iloc[t_id]; ty = y_train1_smote.iloc[t_id]

    vx = X_train1_smote.iloc[v_id]; vy = y_train1_smote.iloc[v_id]

    rf1_2.fit(tx,ty)        

    val_y = rf1_2.predict(vx)

    

    F1_score = f1_score(vy, val_y,average='weighted')

    f1.append(F1_score)

    print(f"fold {fold} f1 {F1_score}")

    print(confusion_matrix(val_y, vy))



print(f"Mean f1 score {np.mean(f1)}")
categorical_features_indices=[0]

from catboost import CatBoostClassifier

rf2_2 = CatBoostClassifier(learning_rate=0.035, 

                          n_estimators=1000, 

                          subsample=0.075, 

                          max_depth=4,

                          l2_leaf_reg = 40,

                          verbose=100,

                          bootstrap_type="Bernoulli",

                          class_weights=[1, 1, 1, 1],

                          loss_function='MultiClass')



kf = StratifiedKFold(n_splits=7,shuffle=True,random_state=99)

f1 = []



for fold,(t_id,v_id) in enumerate(kf.split(X_train2_smote,y_train2_smote)):

    tx = X_train2_smote.iloc[t_id]; ty = y_train2_smote.iloc[t_id]

    vx = X_train2_smote.iloc[v_id]; vy = y_train2_smote.iloc[v_id]

    rf2_2.fit(tx,ty)

           

    val_y = rf2_2.predict(vx)

    F1_score = f1_score(vy, val_y,average='weighted')

    f1.append(F1_score)

    print(f"fold {fold} f1 {F1_score}")

    print(confusion_matrix(val_y, vy))



print(f"Mean f1 score {np.mean(f1)}")
df_test.head()
df_test.info()
test_profile = ProfileReport(df_test,title='Detailed Customer Report')

test_profile.to_widgets()
# Removing Unnecessary Columns

Z = del_col('pet_id',df_test)





# Imputation

Z['condition'] = impute_col(Z['condition'],3.0)

Z['condition'] = Z['condition'].astype('int')



# Standardization - converting cm to mts

Z['length(cm)'] = Z['length(m)'].apply(lambda x: x*100)

Z = del_col('length(m)',Z)

val = Z['length(cm)'].mean()

Z['length(cm)'] = Z['length(cm)'].replace(to_replace=0, value=val)
Z[['issue_date','listing_date']] = Z[['issue_date','listing_date']].apply(pd.to_datetime) #if conversion required

Z['diff_days'] = (Z['listing_date'] - Z['issue_date']).dt.days

Z = del_col('issue_date',Z)

Z = del_col('listing_date',Z)
# Diff days Standardization

Z['diff_days'] = abs(Z['diff_days'])

Z['diff_days'] =np.array(np.array(Z['diff_days']) / 365.)



# Quantile Based Binning

quantile_list = [0, .25, .5, .75, 1.]

quantiles = Z['length(cm)'].quantile(quantile_list)

quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']

Z['length_label'] = pd.qcut(Z['length(cm)'],q=quantile_list, labels=quantile_labels)

Z = pd.concat([Z,pd.get_dummies(Z['condition'], prefix='condition')],axis=1)

Z = pd.concat([Z,pd.get_dummies(Z['X2'], prefix='X2')],axis=1)

Z = pd.concat([Z,pd.get_dummies(Z['X1'], prefix='X1')],axis=1)

Z = pd.concat([Z,pd.get_dummies(Z['color_type'], prefix='color_type')],axis=1)

Z = pd.concat([Z,pd.get_dummies(Z['length_label'], prefix='length_label')],axis=1)



Z = del_col('condition',Z)

Z = del_col('color_type',Z)

Z = del_col('X2',Z)

Z = del_col('X1',Z)

Z = del_col('length(cm)',Z)

Z = del_col('length_label',Z)

Z.head()
# Adding Missing Columns from training Set

Z['color_type_Black Tiger'] = 0

Z['color_type_Brown Tiger'] = 0

Z['X1_3'] = 0

Z['X1_19'] = 0

Z = Z[X1.columns]
breed_category = rf1_2.predict(Z)

breed_category
pet_category = rf2_2.predict(Z)

pet_category
submission = pd.DataFrame(df_test['pet_id'],columns=['pet_id',])

submission['breed_category'] = breed_category

submission['pet_category'] = pet_category
submission['breed_category'].value_counts()
submission['pet_category'].value_counts()
submission.head(10)