import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv('/kaggle/input/titanic/train.csv')

print(train.columns)

test = pd.read_csv('/kaggle/input/titanic/test.csv')

print(test.columns)

# train.drop(['SibSp','Cabin','Embarked'],axis=1,inplace=True)

# test.drop(['SibSp','Cabin','Embarked'],axis=1,inplace=True)

train['last_name'] = train['Name'].apply(lambda x:str(x).split(',')[1].split('.')[1].strip().split(' ')[0])

test['last_name'] = test['Name'].apply(lambda x:str(x).split(',')[1].split('.')[1].strip().split(' ')[0])
def Single_Feature_Test(tr_data,te_data,fea_name,label_name):

    import xgboost as xgb

    ##part1

    if tr_data[fea_name].dtype=='object' or te_data[fea_name].dtype=='object': 

        trans_dict = {}

        for item in list(tr_data[fea_name].values):

            trans_dict[item] = trans_dict.get(item,len(trans_dict))

        for item in list(te_data[fea_name].values):

            trans_dict[item] = trans_dict.get(item,len(trans_dict))

        tr_data[fea_name] = tr_data[fea_name].apply(lambda x : trans_dict[x])

        te_data[fea_name] = te_data[fea_name].apply(lambda x : trans_dict[x])         

    xgb_params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 

                  'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 

                  'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 6, 

                  'min_child_weight': 1, 'missing': None, 'n_estimators': 500, 

                  'n_jobs': 1, 'nthread': -1, 'objective': 'binary:logistic', 

                  'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 

                  'seed': 917, 'silent': False, 'subsample': 1, 'verbosity': 1, 'tree_method': 'auto'}

    xgtrain = xgb.DMatrix(tr_data[[fea_name]], label=tr_data[label_name])

    cvresult_eff = xgb.cv(xgb_params,xgtrain, num_boost_round=100, nfold=5,metrics='auc', seed=1,early_stopping_rounds = 5)

    ##part2

    tr_data['is_train'] = 1

    te_data['is_train'] = 0

    al_data = pd.concat([tr_data[[fea_name,'is_train']],te_data[[fea_name,'is_train']]])

    xgtrain = xgb.DMatrix(al_data[[fea_name]], label=al_data['is_train'])

    cvresult_shift = xgb.cv(xgb_params,xgtrain, num_boost_round=100, nfold=5,metrics='auc', seed=1,early_stopping_rounds = 5)

    fea_eff = '%.3f' % cvresult_eff['test-auc-mean'][cvresult_eff.shape[0]-1]

    fea_shif = '%.3f' % cvresult_shift['test-auc-mean'][cvresult_shift.shape[0]-1]

    if float(fea_shif) >= 0.7:

        fea_eff = 0.0

    print(fea_name,'fea_eff:',fea_eff,'fea_shif:',fea_shif) 

    return fea_eff,fea_shif
def bruce_force_solver(Corr,Eff,Shif):

    all_f_name = Corr.keys().to_list()

    max_sum_v = 0.0

    best_judge = {}

    for x in all_f_name:

        Corr[x][x]=0.0

    for i in range(1,2**len(all_f_name)):

        cod = ('0'*len(all_f_name) + bin(i)[2:])[-len(all_f_name):]

        judge = {}

        for idx,item in enumerate(all_f_name):

            judge[item] = int(cod[idx])

        sum_v = 0

        for x in all_f_name:

            part = 1.0

            num = 0.0

            for y in all_f_name:

                part*=(Eff[y]*(1-abs(Corr[x][y])))**(-judge[y])

                num+=judge[y]

            sum_v += judge[x]*(part**(-1.0/num))

        if sum_v>max_sum_v:

            max_sum_v = sum_v

            best_judge = judge

    return best_judge
def bruce_force_solve(Corr,Eff,Shif):

    all_f_name = Corr.keys().to_list()

    max_sum_v = 0.0

    best_judge = {}

    corr_mat = np.ones((len(all_f_name),len(all_f_name)))

    eff_arr = np.ones((len(all_f_name),len(all_f_name)))

    for x in all_f_name:

        Corr[x][x]=0.0

    for ix,x in enumerate(all_f_name):

        for iy,y in enumerate(all_f_name):

            corr_mat[ix,iy] = 1-Corr[x][y]

            eff_arr[ix,iy] = np.sqrt(Eff[x]*Eff[y])

    for i in range(1,2**len(all_f_name)):

        cod = ('0'*len(all_f_name) + bin(i)[2:])[-len(all_f_name):]

        judge = np.zeros(len(all_f_name))

        for idx,item in enumerate(all_f_name):

            judge[idx] = float(cod[idx])

        judge = np.mat(judge,dtype=np.float)

        sum_v = np.dot(judge,np.sum(np.multiply(corr_mat,eff_arr),axis=1))

        if sum_v>max_sum_v:

            max_sum_v = sum_v

            best_judge = judge  

    bj = {}

    print(best_judge)

    print(best_judge.shape)

    for i in range(len(all_f_name)):

        bj[all_f_name[i]] = best_judge[0,i]

    return bj
def f_transform(df):

    from sklearn import preprocessing

    for fea_name in df.columns.to_list():

        if df[fea_name].dtype=='object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(df[fea_name].values))

            df[fea_name] = lbl.transform(list(df[fea_name].values))

#             df[fea_name+'_label_enc'] = lbl.transform(list(df[fea_name].values))

#             group_obj = tr[[x,y]].groupby(x)[y]

#             count_d = group_obj.count().to_dict()

#             df[fea_name+'_freq_enc'] = df[fea_name].map(count_d)

#         if df[fea_name].dtype!='object':

#             df[fea_name+'_bin'] = pd.qcut(df[fea_name],10)

#             df[fea_name+'_freq_enc'] = (df[fea_name] - df[fea_name].mean())/df[fea_name].std()  ###一般不建议直接变换

#             df[fea_name+'_round'] = df[fea_name].map(lambda x:round(x))

#             df[fea_name+'_dec'] = df[fea_name].map(lambda x:x-int(x))

    return df
def train_cv(tr_data,te_data,label):

    import xgboost as xgb

    from sklearn import preprocessing

    for fea_name in tr_data.columns:

        if tr_data[fea_name].dtype=='object' or te_data[fea_name].dtype=='object': 

            trans_dict = {}

            for item in list(tr_data[fea_name].values):

                trans_dict[item] = trans_dict.get(item,len(trans_dict))

            for item in list(te_data[fea_name].values):

                trans_dict[item] = trans_dict.get(item,len(trans_dict))

            tr_data[fea_name] = tr_data[fea_name].apply(lambda x : trans_dict[x])

            te_data[fea_name] = te_data[fea_name].apply(lambda x : trans_dict[x]) 

#             lbl = preprocessing.LabelEncoder()

#             lbl.fit(list(tr_data[fea_name].values) + list(te_data[fea_name].values))

#             tr_data[fea_name] = lbl.transform(list(tr_data[fea_name].values))

#             te_data[fea_name] = lbl.transform(list(te_data[fea_name].values)) 

    xgb_params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 

                  'colsample_bynode': 1, 'colsample_bytree': 1, 'gamma': 0, 

                  'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 7, 

                  'min_child_weight': 1, 'missing': None, 'n_estimators': 500, 

                  'n_jobs': 8, 'nthread': -1, 'objective': 'binary:logistic', 

                  'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 0, 'scale_pos_weight': 1, 

                  'seed': 42, 'silent': False, 'subsample': 1, 'verbosity': 1, 'tree_method': 'hist'}

    xgtrain = xgb.DMatrix(tr_data, label=label)

    cvresult_eff = xgb.cv(xgb_params,xgtrain, num_boost_round=100, nfold=3,metrics='auc', seed=1,early_stopping_rounds = 5)

    tr_data['is_train'] = 1

    te_data['is_train'] = 0

    al_data = pd.concat([tr_data,te_data])

    xgtrain = xgb.DMatrix(al_data.drop(['is_train'],axis=1), label=al_data['is_train'])

    cvresult_shift = xgb.cv(xgb_params,xgtrain, num_boost_round=100, nfold=3,metrics='auc', seed=1,early_stopping_rounds = 5)

    fea_eff = '%.3f' % cvresult_eff['test-auc-mean'][cvresult_eff.shape[0]-1]

    fea_shif = '%.3f' % cvresult_shift['test-auc-mean'][cvresult_shift.shape[0]-1]

    print('EFF_AUC:',fea_eff)

    print('ADV_AUC:',fea_shif)

    return fea_eff,fea_shif
def correlation_matrix(df,labels):

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm



    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 500)

    cax = ax1.imshow(df, interpolation="nearest", cmap=cmap)

    ax1.grid(True)

    plt.title('Feature Correlation')

#     labels=['Sex','Length','Diam','Height','Whole','Shucked','Viscera','Shell','Rings',]

    ax1.set_xticklabels(labels,fontsize=6)

    ax1.set_yticklabels(labels,fontsize=6)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels

    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])

    plt.show()
def find_best(train,test):

    eff_dict = {}

    shif_dict = {}

    import seaborn as sns

    for fea in test.columns:

        fea_eff,fea_shif = Single_Feature_Test(train.copy(),test.copy(),fea,'Survived')

        eff_dict[fea] = float(fea_eff)

        shif_dict[fea] = float(fea_shif)



    corr_mat = f_transform(train[test.columns]).corr(method='spearman')

#     correlation_matrix(corr_mat,test.columns)

#     sns.heatmap(corr_mat, 

#             xticklabels=corr_mat.columns,

#             yticklabels=corr_mat.columns)

    best_judge = bruce_force_solve(corr_mat,eff_dict,shif_dict)

    print(best_judge)

    feature_list = np.array(list(best_judge.keys()))

    choose_ind = np.where(np.array(list(best_judge.values()))==1)

    drop_ind = np.where(np.array(list(best_judge.values()))==0)

    print('Selected Feature:',feature_list[choose_ind])

    print('Droped Feature:',feature_list[drop_ind])

    return feature_list,choose_ind,eff_dict,corr_mat
def find_top_k_cro(k,feature_list,eff_dict,corr_mat):

    cro_dict = {}

    for idx_i,f_i in enumerate(feature_list):

        for idx_j,f_j in enumerate(feature_list):

            if idx_i<=idx_j or len(set(f_j.split('_'))&set(f_i.split('_')))>0 or (f_i+'_'+f_j) in eff_dict.keys() or len(f_i.split('_'))>1 or len(f_j.split('_'))>1:

                continue

            eff_i_j = (eff_dict[f_i]*eff_dict[f_j])/(1+abs(corr_mat[f_i][f_j]))

            cro_dict[f_i+'**'+f_j] = eff_i_j

    top_arr = pd.DataFrame([],columns=['name','value'])

    top_arr['name'] = list(cro_dict.keys())

    top_arr['value'] = list(cro_dict.values())

    top_arr.sort_values(by='value',ascending=False,inplace=True)     

    return top_arr['name'].head(k).to_list()
def cross_tool(tr,te,x,y):

    print(tr[x].dtype,tr[y].dtype)

    if tr[x].dtype=='object' and tr[y].dtype=='object': 

        tr[x+'_'+y] = tr[x].astype(str)+'_'+tr[y].astype(str)

        te[x+'_'+y] = te[x].astype(str)+'_'+te[y].astype(str)

    if tr[x].dtype=='object' and tr[y].dtype!='object': 

        group_obj = tr[[x,y]].groupby(x)[y]

        mean_d = group_obj.mean().to_dict()

        std_d = group_obj.std().to_dict()

        tr[x+'_'+y] = (tr[y] -  tr[x].map(mean_d))/tr[x].map(std_d)

        te[x+'_'+y] = (te[y] -  te[x].map(mean_d))/te[x].map(std_d)

    if tr[x].dtype!='object' and tr[y].dtype=='object': 

        group_obj = tr[[x,y]].groupby(y)[x]

        mean_d = group_obj.mean().to_dict()

        std_d = group_obj.std().to_dict()

        tr[x+'_'+y] = (tr[x] -  tr[y].map(mean_d))/tr[y].map(std_d)

        te[x+'_'+y] = (te[x] -  te[y].map(mean_d))/te[y].map(std_d)

    if tr[x].dtype!='object' and tr[y].dtype!='object': 

        tr[x+'_'+y] = tr[x].astype(float)*tr[y].astype(float)

        te[x+'_'+y] = te[x].astype(float)*te[y].astype(float)

    return tr,te




old_train = train.copy(deep=True)

old_test = test.copy(deep=True)

old_list = old_test.columns

old_choosen_f=[]

fuz_auc_stat = []

log_auc_stat = []



for i in range(4):

    feature_list,choose_ind,eff_dict,corr_mat = find_best(train,test)

    train = train[np.append(feature_list[choose_ind],'Survived')]

    test = test[feature_list[choose_ind]]

    best_cro = find_top_k_cro(2,feature_list[choose_ind],eff_dict,corr_mat)

    choosen_f = feature_list[choose_ind]

#     if set(choosen_f) == set(old_choosen_f):

#         break

    for item in best_cro:

        x = item.split('**')[0]

        y = item.split('**')[1]

#         if train[x].dtype!='object' and train[y].dtype!='object':

#             continue

        train,test = cross_tool(train,test,x,y)

        old_train[x+'_'+y] = train[x+'_'+y]

        old_test[x+'_'+y] = test[x+'_'+y]

#         train[x+'_'+y] = train[x].astype(str)+'_'+train[y].astype(str)

#         test[x+'_'+y] = test[x].astype(str)+'_'+test[y].astype(str)

        fea_eff,fea_shif = Single_Feature_Test(train.copy(),test.copy(),x+'_'+y,'Survived')

        if float(fea_shif)>=0.6:

            continue

        ori_set = list(old_list)

        print(ori_set)

        a,b = train_cv(old_train[ori_set],old_test[ori_set],train['Survived'])

        ori_set.append(x+'_'+y)

        print(ori_set)

        a,b = train_cv(old_train[ori_set],old_test[ori_set],train['Survived'])

        fuz_auc_stat.append(float(a))

        ori_set = list(choosen_f)

        print(ori_set)

        a,b = train_cv(train[ori_set],test[ori_set],train['Survived'])

        ori_set.append(x+'_'+y)

        print(ori_set)

        a,b = train_cv(train[ori_set],test[ori_set],train['Survived'])

        log_auc_stat.append(float(a))

        

        feature_list = np.append(feature_list,x+'_'+y)

        old_list = np.append(old_list,x+'_'+y)

        old_choosen_f = choosen_f

        choosen_f = np.append(choosen_f,x+'_'+y)
from matplotlib import pyplot as plt

print(log_auc_stat)

print(fuz_auc_stat)

plt.plot(range(len(log_auc_stat)),log_auc_stat)

plt.plot(range(len(log_auc_stat)),fuz_auc_stat)

plt.show()