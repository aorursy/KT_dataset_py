import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split,StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,roc_auc_score,roc_curve, auc

from imblearn.over_sampling import RandomOverSampler

import lightgbm as lgb
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
traindata=pd.read_csv("/kaggle/input/amexpert2019/train.csv")

testdata=pd.read_csv("/kaggle/input/amexpert2019/test.csv")

campaigndata=pd.read_csv("/kaggle/input/amexpert2019/campaign_data.csv")

coupon_item_mappingdata=pd.read_csv("/kaggle/input/amexpert2019/coupon_item_mapping.csv")

cust_demodata=pd.read_csv("/kaggle/input/amexpert2019/customer_demographics.csv")

itemdata=pd.read_csv("/kaggle/input/amexpert2019/item_data.csv")

cust_trandata=pd.read_csv("/kaggle/input/amexpert2019/customer_transaction_data.csv")
print('Train Data Shape ---',traindata.shape)

print('Test Data Shape ----', testdata.shape)

print('Campaign Data Shape ----', campaigndata.shape)

print('Coupon Item Mapping Shape ----', coupon_item_mappingdata.shape)

print('Customer Demographic Shape ----', cust_demodata.shape)

print('Item Data Shape ----', itemdata.shape)

print('Customer Transaction Data Shape ----', cust_trandata.shape)

print(' Train Columns :' ,traindata.columns)

print('Campaign Data columns :', campaigndata.columns)

print('Coupon Item Mapping columns ----', coupon_item_mappingdata.columns)

print('Customer Demographic columns ----', cust_demodata.columns)

print('Item Data columns ----', itemdata.columns)

print('Customer Transaction Data columns ----', cust_trandata.columns)
campaigndata.head()
def BuildDataSet(initialDataSet):

  campaign=pd.merge(initialDataSet, campaigndata, how='left', on=['campaign_id'])

  campaign.drop_duplicates(inplace=True)



  _campaign_cust=pd.merge(campaign, cust_demodata, how='left', on=['customer_id'])

  _campaign_cust.drop_duplicates(inplace=True)



  customer_data = (cust_trandata.groupby('customer_id').agg({  # 'date':['min'],

                                                              'quantity':['mean'],

                                                              'selling_price':['mean'],

                                                              'other_discount':['mean'],

                                                              'coupon_discount':['mean']

                                                              })).reset_index()

  customer_data.columns = ['customer_id','quantity','selling_price','other_discount','coupon_discount']

  # print(customer_data.head())

  customer_data.drop_duplicates(inplace=True)



  _campaign_cust_tran=pd.merge(_campaign_cust, customer_data, how='left', on=['customer_id'])

  _campaign_cust_tran.drop_duplicates(inplace=True)

  # print(_campaign_cust_tran.columns)



  coupon_item_data=pd.merge(coupon_item_mappingdata, itemdata, how='left', on=['item_id'])



  coupon_item_data=coupon_item_data.groupby('coupon_id').agg({

#         'item_id':['mean'],

        'brand':['mean'],

        'brand_type':['max'],

        'category':['max']

  }).reset_index()

  coupon_item_data.columns = ['coupon_id','brand','brand_type','category'] 

#     'item_id'

  # coupon_item_data.head()

  coupon_item_data.drop_duplicates(inplace=True)



  _campaign_cust_tran_coupon=pd.merge(_campaign_cust_tran, coupon_item_data, how='left', on=['coupon_id'])

  _campaign_cust_tran_coupon.drop_duplicates(inplace=True)



  # coupon_data = (coupon_item_data.groupby('coupon_id').agg({'item_id': ['max']})).reset_index()

  # coupon_data.columns = ['coupon_id','item_id']

  # coupon_data.drop_duplicates(inplace=True)

  # # print(coupon_data.columns)



  # _campaign_cust_coupon_tran=pd.merge(_campaign_cust_tran, coupon_data, how='left', on=['coupon_id'])

  # _campaign_cust_coupon_tran.drop_duplicates(inplace=True)

  # # print(_campaign_cust_coupon_tran.columns)



  # _campaign_cust_coupon_tran['item_id']=_campaign_cust_coupon_tran['item_id_x']



  # _campaign_cust_coupon_trans_item=pd.merge(_campaign_cust_coupon_tran, itemdata, how='left', on=['item_id','item_id'])

  # _campaign_cust_coupon_trans_item.drop_duplicates(inplace=True)

  # print(_campaign_cust_coupon_trans_item.head())



  return _campaign_cust_tran_coupon
finaltrain=BuildDataSet(traindata)

print(finaltrain.shape)

finaltest=BuildDataSet(testdata)

print(finaltest.shape)
finaltrain.isnull().sum()
dict={}

def GetUniqueValues(df):

  dict={}

  for i in df.columns:

    dict[i]=df[i].unique()

  return dict



GetUniqueValues(finaltrain)
# #Treat Missing values

finaltrain['age_range'] = finaltrain['age_range'].fillna(finaltrain['age_range'].mode()[0])

finaltrain['marital_status'] = finaltrain['marital_status'].fillna(finaltrain['marital_status'].mode()[0])

finaltrain['rented'] = finaltrain['rented'].fillna(finaltrain['rented'].mode()[0])

finaltrain['family_size'] = finaltrain['family_size'].fillna(finaltrain['family_size'].mode()[0])

finaltrain['no_of_children'] = finaltrain['no_of_children'].fillna(finaltrain['no_of_children'].mode()[0])

finaltrain['income_bracket'] = finaltrain['income_bracket'].fillna(finaltrain['income_bracket'].mean())



finaltest['age_range'] = finaltest['age_range'].fillna(finaltest['age_range'].mode()[0])

finaltest['marital_status'] = finaltest['marital_status'].fillna(finaltest['marital_status'].mode()[0])

finaltest['rented'] = finaltest['rented'].fillna(finaltest['rented'].mode()[0])

finaltest['family_size'] = finaltest['family_size'].fillna(finaltest['family_size'].mode()[0])

finaltest['no_of_children'] = finaltest['no_of_children'].fillna(finaltest['no_of_children'].mode()[0])

finaltest['income_bracket'] = finaltest['income_bracket'].fillna(finaltest['income_bracket'].mean())
# From boxplot we can observe their is an outlier for quantity  above 460 its an outlier so do capping



finaltrain['quantity'][finaltrain['quantity']>=460]=460

finaltest['quantity'][finaltest['quantity']>=460]=460





# From boxplot we can observe their is an outlier for selling_price  above 176 its an outlier so do capping & flooring



finaltrain['selling_price'][finaltrain['selling_price']>=176]=176

finaltest['selling_price'][finaltest['selling_price']>=176]=176



finaltest['selling_price'][finaltest['selling_price']<=65]=65



# From boxplot we can observe their is an outlier for other_discount below -30 its an outlier



finaltrain['other_discount'][finaltrain['other_discount']<=-29]=-30

finaltest['other_discount'][finaltest['other_discount']<=-29]=-30



# from boxplot we can observe their is an outlier for coupon_discount below



finaltrain['coupon_discount'][finaltrain['coupon_discount']<=-1.5]=-1.5

finaltest['coupon_discount'][finaltest['coupon_discount']<=-1.5]=-1.5



# from boxplot we can observe their is an outlier for brand below

finaltrain['brand'][finaltrain['brand']>=2416]=2416

finaltest['brand'][finaltest['brand']>=2416]=2416



import sys

np.set_printoptions(threshold=sys.maxsize)

np.percentile(finaltest['selling_price'], [  80,85,86,87,88,89,90,95,100])

                                       

# 0.,   5.,  10.,  15.,  20.,  25.,  30.,  35.,  40.,  45.,  50.,55.,  60.,  65.,  70.,  75.,  80.,  85.,  90.,  95., 100.]
# sns.boxplot(finaltrain['item_id'])

# plt.show()
#Creating new feature

finaltrain['start_date'] = pd.to_datetime(finaltrain['start_date'])

finaltrain['end_date'] = pd.to_datetime(finaltrain['end_date'])



finaltest['start_date'] = pd.to_datetime(finaltest['start_date'])

finaltest['end_date'] = pd.to_datetime(finaltest['end_date'])





finaltrain['end_date_month'] = finaltrain['end_date'].dt.month

finaltrain['end_date_dayofweek'] = finaltrain['end_date'].dt.dayofweek 

finaltrain['end_date_dayofyear'] = finaltrain['end_date'].dt.dayofyear 

finaltrain['end_date_days_in_month'] = finaltrain['end_date'].dt.days_in_month 

finaltrain['start_date_month'] = finaltrain['start_date'].dt.month

finaltrain['start_date_dayofweek'] = finaltrain['start_date'].dt.dayofweek 

finaltrain['start_date_dayofyear'] = finaltrain['start_date'].dt.dayofyear 

finaltrain['start_date_days_in_month'] = finaltrain['start_date'].dt.days_in_month

finaltrain['diff_dayofweek'] = finaltrain['end_date_dayofweek'] - finaltrain['start_date_dayofweek']

finaltrain['diff_dayofyear'] = finaltrain['end_date_dayofyear'] - finaltrain['start_date_dayofyear']



finaltest['end_date_month'] = finaltest['end_date'].dt.month

finaltest['end_date_dayofweek'] = finaltest['end_date'].dt.dayofweek 

finaltest['end_date_dayofyear'] = finaltest['end_date'].dt.dayofyear 

finaltest['end_date_days_in_month'] = finaltest['end_date'].dt.days_in_month 

finaltest['start_date_month'] = finaltest['start_date'].dt.month

finaltest['start_date_dayofweek'] = finaltest['start_date'].dt.dayofweek 

finaltest['start_date_dayofyear'] = finaltest['start_date'].dt.dayofyear 

finaltest['start_date_days_in_month'] = finaltest['start_date'].dt.days_in_month

finaltest['diff_dayofweek'] = finaltest['end_date_dayofweek'] - finaltest['start_date_dayofweek']

finaltest['diff_dayofyear'] = finaltest['end_date_dayofyear'] - finaltest['start_date_dayofyear']





finaltrain['campaign_days']=((pd.to_datetime(finaltrain['end_date'], format='%d/%m/%y')) - 

                             (pd.to_datetime(finaltrain['start_date'], format='%d/%m/%y'))).dt.days



finaltest['campaign_days']=((pd.to_datetime(finaltest['end_date'], format='%d/%m/%y')) - 

                             (pd.to_datetime(finaltest['start_date'], format='%d/%m/%y'))).dt.days
finaltrain.describe(include='all')
corr=finaltrain.corr()**2

sns.heatmap(corr)

plt.show()
# sns.pairplot(finaltrain, hue='redemption_status')

# plt.show()
sns.countplot('campaign_type', data=finaltrain)

plt.show()
# check based on campaign Type , will cusotmers redemed ?. Yes X campaign customers redemeed

sns.countplot('campaign_type', hue='redemption_status', data=finaltrain)
# it saying imbalanced data set

sns.countplot('redemption_status', data=finaltrain)
# check with age group redeemed 

sns.countplot('age_range',hue='redemption_status', data=finaltrain)
# check with age group redeemed 

sns.catplot(x='age_range',hue='redemption_status', col='family_size',row='no_of_children',data=finaltrain, kind='count')
#mostly redemption used in Grocery only

sns.catplot(x='redemption_status', hue='category', data=finaltrain, kind='count', palette='Set1')
def HandleCategoryFeatures(traindf, testdf):    



    # Handling Category features



    traindf['age_range']=traindf['age_range'].replace('46-55',46)

    traindf['age_range']=traindf['age_range'].replace('36-45',36)

    traindf['age_range']=traindf['age_range'].replace('18-25',18)

    traindf['age_range']=traindf['age_range'].replace('26-35',26)

    traindf['age_range']=traindf['age_range'].replace('56-70',56)

    traindf['age_range']=traindf['age_range'].replace('70+',70)



    traindf['family_size']=traindf['family_size'].replace('5+',5)



    traindf['no_of_children']=traindf['no_of_children'].replace('3+',3)



    traindf['no_of_children']=traindf['no_of_children'].astype(int)

    traindf['family_size']=traindf['family_size'].astype(int)

    traindf['age_range']=traindf['age_range'].astype(int)

    traindf['rented']=traindf['rented'].astype(int)    



    testdf['age_range']=testdf['age_range'].replace('46-55',46)

    testdf['age_range']=testdf['age_range'].replace('36-45',36)

    testdf['age_range']=testdf['age_range'].replace('18-25',18)

    testdf['age_range']=testdf['age_range'].replace('26-35',26)

    testdf['age_range']=testdf['age_range'].replace('56-70',56)

    testdf['age_range']=testdf['age_range'].replace('70+',70)



    testdf['family_size']=testdf['family_size'].replace('5+',5)

    testdf['no_of_children']=testdf['no_of_children'].replace('3+',3)





    testdf['no_of_children']=testdf['no_of_children'].astype(int)

    testdf['family_size']=testdf['family_size'].astype(int)

    testdf['age_range']=testdf['age_range'].astype(int)

    testdf['rented']=testdf['rented'].astype(int)         
feature={}

final_pred_test_full =0

def MOdelExecution(model):

    

    kf = StratifiedKFold(n_splits=2,shuffle=True,random_state=45)

    pred_test_full=0

    cv_score =[]

    TrainScore=[]

    Testscore=[]

    F1Score=[]

    Sensitivity=[]

    Specificity=[]

    Auc=[]

    i=1

    feature_importances=[]

    for train_index,test_index in kf.split(X,Y):

        print('{} of KFold {}'.format(i,kf.n_splits))

        xtr,xvl = X.loc[train_index],X.loc[test_index]

        ytr,yvl = Y.loc[train_index],Y.loc[test_index]

        

        mod=model

        mod.fit(xtr,ytr)

        pred=mod.predict(xvl)

        

        rocscore = roc_auc_score(yvl,mod.predict(xvl))

        cv_score.append(rocscore)  

        

        Score=mod.score(xtr,ytr)

        

        TrainScore.append(Score)

        

        Testaccuracy=accuracy_score(yvl,mod.predict(xvl))

        

        Testscore.append(Testaccuracy)



        confusion=confusion_matrix(yvl, pred)



        y_pred_quant = mod.predict_proba(xvl)[:, 1]



        f1=f1_score(yvl, pred)        

        F1Score.append(f1)



        total=sum(sum(confusion))



        sensitivity = confusion[0,0]/(confusion[0,0]+confusion[1,0])

        

        Sensitivity.append(sensitivity)



        specificity = confusion[1,1]/(confusion[1,1]+confusion[0,1])

        

        Specificity.append(specificity)



        fpr, tpr, thresholds = roc_curve(yvl, y_pred_quant)

        

        Auc.append(auc(fpr,tpr))

        

        pred_test_full +=y_pred_quant

        i+=1

    final_pred_test_full=pred_test_full

    print('Cv_Score',cv_score)

    print('TrainScore', TrainScore)

    print('TestScore', Testscore)

    print('F1 Score',F1Score)

    print('Sensitivity',Sensitivity)

    print('Specificity',Specificity)    
finaltrain2=finaltrain.copy()

finaltest2=finaltest.copy()



HandleCategoryFeatures(finaltrain2,finaltest2)

finaltrain2=pd.get_dummies(data=finaltrain2, columns=['marital_status', 'campaign_type','brand_type','category'], drop_first=True)



finaltest2=pd.get_dummies(data=finaltest2, columns=['marital_status', 'campaign_type','brand_type','category'], drop_first=True)



#Drop unnecessary columns

finaltrain2.drop(finaltrain2[['id','start_date','end_date']], axis=1, inplace=True)

finaltest2.drop(finaltest2[['start_date','end_date']], axis=1, inplace=True)

# finaltest_oversampling.columns

# 1) Over Sampling

os=RandomOverSampler(ratio=1)

X_train_res, y_train_res=os.fit_sample(finaltrain2.drop(['redemption_status'],axis=1), finaltrain2['redemption_status'])



OverSampleX=pd.DataFrame(X_train_res, columns=finaltrain2.drop('redemption_status', axis=1).columns)

OverSampleY=pd.DataFrame(y_train_res, columns=finaltrain2[['redemption_status']].columns)

finaltrain_oversampling=''

finaltrain_oversampling = pd.concat([OverSampleX, OverSampleY], axis=1)

finaltest_oversampling=''

finaltest_oversampling=finaltest2.copy()

X=finaltrain_oversampling.drop(['redemption_status'], axis=1)

Y=(finaltrain_oversampling[['redemption_status']])



s_scaler = StandardScaler()

df_s = s_scaler.fit_transform(X)

X = pd.DataFrame(df_s, columns=finaltrain_oversampling.drop(['redemption_status'], axis=1).columns)

mod=lgb.LGBMClassifier(boosting_type= 'gbdt', objective= 'binary', metric='auc', bagging_freq=1, subsample=1, feature_fraction= 0.7,

              num_leaves= 8, learning_rate= 0.05, lambda_l1=5,max_bin=255)



MOdelExecution(mod)
# mod.fit(X_Train, Y_Train)

finaltest_new=finaltest_oversampling.copy()



s_scaler = StandardScaler()



df_s = s_scaler.fit_transform(finaltest_new.drop(['id'], axis=1))



X2_Test = pd.DataFrame(df_s, columns=finaltest_new.drop(['id'], axis=1).columns)

X2_Test['id']=finaltest_oversampling['id']

X2_Test['category_Restauarant']=0



Test_Pred=mod.predict(X2_Test.drop(['id'], axis=1))



Test_Prob=mod.predict_proba(X2_Test.drop(['id'], axis=1))[:,1]

finaltest_new['redemption_Prob']=Test_Prob

finaltest_new['redemption_status']=Test_Pred



print(finaltest_new['redemption_status'].value_counts())



print(finaltest_new['redemption_Prob'].value_counts())



finaltest_new['redemption_Prob'] = finaltest_new['redemption_Prob'].apply(lambda x: 1 if x>0.5 else 0)



print(finaltest_new['redemption_Prob'].value_counts())



testdv=finaltest_new[['id','redemption_status']]

testdv.to_csv('final_submission_05_LGBM.csv',index=False)