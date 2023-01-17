import pandas as pd

import math

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sn

from pandas.plotting import scatter_matrix

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

%matplotlib inline

## denote the working directory
def in_range (lowerlimit, upperlimit, value):

    return lowerlimit <= value <= upperlimit



def calculate_scores(h1max,h1min,d1max,d1min,value_ranges):

    max_score=0

    min_score=0

    hd_value=pd.Series([h1max,h1min,d1max,d1min]).dropna()

    if hd_value.empty:

        return np.nan

    else:

        max_value=max(hd_value)

        min_value=min(hd_value)

        for value_range in value_ranges:

            if in_range(value_range[0], value_range[1], max_value):

                max_score=int(value_range[2])

                break

        for value_range in value_ranges: 

            if in_range(value_range[0],value_range[1], min_value):

                min_score=int(value_range[2])

                break

        if max_score>=min_score:

            return max_score

        else:

            return min_score 



def calculate_value(h1max,h1min,d1max,d1min,value_ranges):

    max_score=0

    min_score=0

    hd_value=pd.Series([h1max,h1min,d1max,d1min]).dropna()

    if hd_value.empty:

        return np.nan

    else:

        max_value=max(hd_value)

        min_value=min(hd_value)

        for value_range in value_ranges:

            if in_range(value_range[0], value_range[1], max_value):

                max_score=int(value_range[2])

                break

        for value_range in value_ranges: 

            if in_range(value_range[0],value_range[1], min_value):

                min_score=int(value_range[2])

                break

        if max_score>=min_score:

            return max_value

        else:

            return min_value 



apache_range={'temp':([41,50,4],[39,40.9,3],[38.5,38.9,1],[36,38.4,0],[34,35.9,1],[32,33.9,2],[30,31.9,3],[19,29.9,4]),

             'heartrate':([180,300,4],[140,179,3],[110,139,2],[70,109,0],[55,69,2],[40,54,3],[0,39,4]),

             'sodium':([180,300,4],[160,179,3],[155,159,2],[150,154,1],[130,149,0],[120,129,2],[111,119,3],[50,110,4]),

             'potassium':([7,9,4],[6,6.9,3],[5.5,5.9,1],[3.5,5.4,0],[3,3.4,1],[2.5,2.9,2],[2.5,2,4]),

             'creatinine':([3.5,30,4],[2,3.4,3],[1.5,1.9,2],[0.6,1.4,0],[0,0.6,2]),

             'hematocrit':([60,100,4],[50,59.9,2],[46,49.9,1],[30,45.9,0],[20,29.9,2],[0,20,4]),

             'wbc':([40,100,4],[20,39.9,2],[15,19.9,1],[3,14.9,0],[1,2.9,2],[0,1,4]),

             'ph':([7.7,9.0,4],[7.6,7.69,3],[7.5,7.59,1],[7.33,7.49,0],[7.25,7.32,2],[7.15,7.24,3],[5,7.15,4]),

             'resprate':([50,100,4],[35,49,3],[25,34,2],[12,24,0],[10,11,1],[6,9,2],[0,5,4])}
df=pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')

Test_df=pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')

All_df=df.append(Test_df)

All_y=All_df['hospital_death']#get y
import plotly.express as px

fig = px.histogram(df[['age','gender','hospital_death','bmi']].dropna(), x="age", y="hospital_death", color="gender",

                   marginal="box", # or violin, rug

                   hover_data=df[['age','gender','hospital_death','bmi']].columns)

fig.show()
age_death_F=df[df['gender']=='F'][['age','hospital_death']].groupby('age').mean().reset_index()

age_death_M=df[df['gender']=='M'][['age','hospital_death']].groupby('age').mean().reset_index()

from plotly.subplots import make_subplots

fig = make_subplots()

fig.add_trace(

    go.Scatter(x=age_death_F['age'], y=age_death_F['hospital_death'], name="Female patients"))

fig.add_trace(

    go.Scatter(x=age_death_M['age'], y=age_death_M['hospital_death'],name="Male patients"))

fig.update_layout(

    title_text="<b>Average hospital death probability of patients<b>")

fig.update_xaxes(title_text="<b>patient age<b>")

fig.update_yaxes(title_text="<b>Average Hospital Death</b>", secondary_y=False)

fig.show()
weight_df=df[['weight','hospital_death','bmi']]

weight_df['weight']=weight_df['weight'].round(0)

weight_df['bmi']=weight_df['bmi'].round(0)

weight_death=weight_df[['weight','hospital_death']].groupby('weight').mean().reset_index()

bmi_death=weight_df[['bmi','hospital_death']].groupby('bmi').mean().reset_index()

fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

fig.add_trace(

    go.Scatter(x=weight_death['weight'], y=weight_death['hospital_death'], name="Weight"),

   row=1, col=1

)

fig.add_trace(

    go.Scatter(x=bmi_death['bmi'], y=bmi_death['hospital_death'], name="BMI"),

    row=1, col=2

)

fig.update_layout(

    title_text="<b>impacts of BMI and weight over patients<b>"

)

fig.update_yaxes(title_text="<b>Average Hospital Death")

fig.show()
ICU_type=df[['icu_type','age','hospital_death']]

ICU_type['icu_type']=ICU_type['icu_type'].replace({'CTICU':'CCU-CTICU',

                                              'Cardiac ICU':'CCT-CTICU',

                                              'CTICU':'CCT-CTICU',

                                              'CSICU':'SICU'})

#ICU_type['pre_icu_los_days']=ICU_type['pre_icu_los_days'].round(0)

ICU_df=ICU_type.groupby(['icu_type','age']).mean().reset_index()

ICU_df['count']=ICU_type.groupby(['icu_type','age']).count().reset_index()['hospital_death']



fig = px.scatter(ICU_df, x="age", y="hospital_death", size="count", color="icu_type",

           hover_name="icu_type", log_x=False, size_max=60,)

fig.update_layout(

    title_text="<b>Survival rate at different types of ICU<b>"

)

fig.update_yaxes(title_text="<b>Average Hospital Death<b>")

fig.update_xaxes(title_text="<b>Age<b>")

fig.show()
ICU_day=df[df['pre_icu_los_days']>=0][['icu_type','pre_icu_los_days','hospital_death']]

ICU_day['icu_type']=ICU_type['icu_type'].replace({'CTICU':'CCU-CTICU',

                                              'Cardiac ICU':'CCT-CTICU',

                                              'CTICU':'CCT-CTICU',

                                              'CSICU':'SICU'})

ICU_day['pre_icu_los_days']=ICU_day['pre_icu_los_days'].round(0)

ICU_df=ICU_day.groupby(['icu_type','pre_icu_los_days']).mean().reset_index()

ICU_df['count']=ICU_day.groupby(['icu_type','pre_icu_los_days']).sum().reset_index()['hospital_death']



fig = px.scatter(ICU_df, x="pre_icu_los_days", y="hospital_death", size="count", color="icu_type",

           hover_name="icu_type", log_x=True, size_max=200,)

fig.update_layout(

    title_text="<b>Survival rate at different length of stay before ICU admission<b>"

)

fig.update_yaxes(title_text="<b>Average Hospital Death<b>")

fig.update_xaxes(title_text="<b>The length of stay of the patient between hospital admission and unit admission <b>")

fig.show()
apache3=df[['age','apache_3j_bodysystem','hospital_death']]

apache3=apache3.groupby(['apache_3j_bodysystem','age']).agg(['size','mean']).reset_index()



apache3['size']=apache3['hospital_death']['size']

apache3['mean']=apache3['hospital_death']['mean']



apache3.drop('hospital_death',axis=1,inplace=True)



systems =list(apache3['apache_3j_bodysystem'].unique())

data = []

list_updatemenus = []

for n, s in enumerate(systems):

    visible = [False] * len(systems)

    visible[n] = True

    temp_dict = dict(label = str(s),

                 method = 'update',

                 args = [{'visible': visible},

                         {'title': '<b>'+s+'<b>'}])

    list_updatemenus.append(temp_dict)

    



for s in systems:

    mask = (apache3['apache_3j_bodysystem'].values == s) 

    trace = (dict(visible = False,     

        x = apache3.loc[mask, 'age'],

        y = apache3.loc[mask, 'mean'],

        mode = 'markers',

        marker = {'size':apache3.loc[mask, 'size']/apache3.loc[mask,'size'].sum()*1000,

                 'color':apache3.loc[mask, 'mean'],

                 'showscale': True})

                   )

    data.append(trace)



data[0]['visible'] = True    

    

layout = dict(updatemenus=list([dict(buttons= list_updatemenus)]),

              xaxis=dict(title = '<b>Age<b>', range=[min(apache3.loc[:, 'age'])-10, max(apache3.loc[:, 'age']) + 10]),

              yaxis=dict(title = '<b>Average Hospital Death<b>', range=[min(apache3.loc[:, 'mean'])-0.1, max(apache3.loc[:, 'mean'])+0.1]),

              title='<b>Survival Rate<b>' )

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='update_dropdown')
All_df['icu_type']=All_df['icu_type'].replace({'CTICU':'CCU-CTICU',

                                              'Cardiac ICU':'CCT-CTICU',

                                              'CTICU':'CCT-CTICU',

                                              'CSICU':'SICU'})



All_df['hospital_admit_source']=All_df['hospital_admit_source'].replace({

                                        'Other ICU':"ICU",'ICU to SDU':"SDU",

                                       'Step-Down Unit (SDU)':"SDU",

                                      'Acute Care/Floor':"Floor",

                                      'Other Hospital':"Other"})



All_df.drop(['encounter_id','patient_id','readmission_status','hospital_death','hospital_id','icu_id','apache_2_bodysystem','apache_3j_bodysystem'],axis=1,inplace=True)



All_df['bmi']=All_df['weight']*10000/(All_df['height']*All_df['height'])

All_df.loc[All_df['bmi']>df['bmi'].max(),'bmi']=df['bmi'].max()

All_df.loc[All_df['bmi']<df['bmi'].min(),'bmi']=df['bmi'].min()



binary=[col for col in All_df.columns if All_df[col].nunique() == 2 and All_df[col].dtypes !='object']

categorical = [col for col in All_df.columns if All_df[col].dtypes == 'object']

All_df['apache_3j_diagnosis']=All_df['apache_3j_diagnosis'].fillna(0).astype(np.int16)

All_df['apache_2_diagnosis']=All_df['apache_2_diagnosis'].fillna(0).astype(np.int16)

categorical.append('apache_2_diagnosis')

categorical.append('apache_3j_diagnosis')



#Binary:we will labelencode Missing as 0, No as 1, Yes as 2 

for col in binary:

    All_df[col]=All_df[col]+1

    All_df[col].fillna(0,inplace=True)

    All_df[col]=All_df[col].astype(np.int8).astype('category')



#STR type Categorical:label encode category

from sklearn import preprocessing

for col in categorical:

    All_df[col] = All_df[col].astype('str')  

    le = preprocessing.LabelEncoder().fit(

            np.unique(All_df[col].unique().tolist()))

    All_df[col] = le.transform(All_df[col])+1

    All_df[col] = All_df[col].replace(np.nan, 0).astype(np.int16).astype('category')



category=categorical+binary
for c in ['resprate','heartrate','sodium']:

    All_df['d1_'+c+'_min']=All_df['d1_'+c+'_min'].round(0)

    All_df['d1_'+c+'_max']=All_df['d1_'+c+'_max'].round(0)

for c in ['temp','potassium','creatinine','hematocrit','wbc']:

    All_df['d1_'+c+'_min']=All_df['d1_'+c+'_min'].round(1)

    All_df['d1_'+c+'_max']=All_df['d1_'+c+'_max'].round(1)

All_df['d1_arterial_ph_min']=All_df['d1_arterial_ph_min'].round(2)

All_df['d1_arterial_ph_max']=All_df['d1_arterial_ph_max'].round(2)

apache=pd.DataFrame()

for c in apache_range.keys():

    if c !='ph':

        apache[c+'_apache']=All_df.apply(lambda row: calculate_value(row['h1_'+c+'_max'],row['h1_'+c+'_min'],row['d1_'+c+'_max'],row['d1_'+c+'_min'],apache_range[c]),axis=1)

    else:

        apache[c+'_apache']=All_df.apply(lambda row:calculate_value(row['h1_arterial_ph_max'],row['h1_arterial_ph_min'],row['d1_arterial_ph_max'],row['d1_arterial_ph_min'],apache_range['ph']),axis=1)

apache_original=set([c for c in All_df.columns.tolist() if '_apache' in c]).intersection(set(apache.columns.tolist()))

other_apache=set(apache.columns.tolist())-apache_original

for c in list(apache_original):

    All_df[c].fillna(apache[c],inplace=True)

for c in list(other_apache):

    if c!='heartrate_apache':

        All_df[c]=apache[c]

    else:

        All_df['heart_rate_apache'].fillna(apache[c],inplace=True)
All_df['hco3pco2_ratio']=All_df['d1_hco3_min']/All_df['d1_arterial_pco2_max']

All_df['map_apache'].fillna((2*All_df['d1_diasbp_min']+All_df['d1_sysbp_max']),inplace=True)
All_df['gcs_sum']=All_df['gcs_eyes_apache']+All_df['gcs_motor_apache']+All_df['gcs_verbal_apache']
invasive_col=[s for s in All_df.columns.tolist() if "invasive" in s]

All_df.drop(invasive_col,axis=1,inplace=True)
feature_dict=pd.read_csv('/kaggle/input/widsdatathon2020/WiDS Datathon 2020 Dictionary.csv')

lab_feature=feature_dict[feature_dict['Category']=='labs']['Variable Name'].tolist()

lab_feature=list(set(lab_feature)-set(invasive_col))

vital_feature=feature_dict[feature_dict['Category']=='vitals']['Variable Name'].tolist()

vital_feature=list(set(vital_feature)-set(invasive_col))

LBG_feature=feature_dict[feature_dict['Category']=='labs blood gas']['Variable Name'].tolist()

LBG_feature=list(set(LBG_feature)-set(invasive_col))

apache_feature=feature_dict[feature_dict['Category']=='APACHE covariate']['Variable Name'].tolist()

col=All_df.columns.tolist()

h1_col=[s for s in col if "h1_" in s]

d1_col=[s for s in col if "d1_" in s]

All_df['null_sum']=All_df.transpose().isnull().sum()

All_df['h1_null_sum']=All_df[h1_col].transpose().isnull().sum()

All_df['d1_null_sum']=All_df[d1_col].transpose().isnull().sum()

All_df['lab_null_sum']=All_df[lab_feature].transpose().isnull().sum()

All_df['vital_null_sum']=All_df[vital_feature].transpose().isnull().sum()

All_df['LBG_null_sum']=All_df[LBG_feature].transpose().isnull().sum()

All_df['apache_null_sum']=All_df[apache_feature].transpose().isnull().sum()
h1_col=[s for s in col if "h1_" in s]

d1_col=[s for s in col if "d1_" in s]

h1d1=list(pd.Series(h1_col).str.replace("h1_",""))

for c in h1d1:

    All_df['diff_'+c]=All_df['d1_'+c]-All_df['h1_'+c]
#RTS = 0.9368 GCS + 0.7326 SBP + 0.2908 RR

RTS_range={'gcs':([13,15,4],[9,12,3],[6,8,2],[4,5,1],[0,3,0]),

           'sysbp':([90,300,4],[76,89,3],[50,75,2],[1,49,1],[0,0,0]),

           'resprate':([10,29,4],[30,150,3],[6,9,2],[1,5,1],[0,0,0])}

RTS=list(RTS_range.keys())

RTS_df=pd.DataFrame()

for c in RTS:

    if c !='gcs':

        RTS_df[c+'_RTS_score']=All_df.apply(lambda row: calculate_scores(row['h1_'+c+'_max'],row['h1_'+c+'_min'],row['d1_'+c+'_max'],row['d1_'+c+'_min'],RTS_range[c]),axis=1)

    else:

        RTS_df['gcs_RTS_score']=All_df.apply(lambda row: calculate_scores(row['gcs_sum'],row['gcs_sum'],row['gcs_sum'],row['gcs_sum'],RTS_range[c]),axis=1)

All_df['RTS_score']=0.9358*RTS_df['gcs_RTS_score']+0.7326*RTS_df['sysbp_RTS_score']+0.2908*RTS_df['resprate_RTS_score']
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split

from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score,explained_variance_score



import lightgbm as lgbm



from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

import warnings





warnings.filterwarnings('ignore')
All_impute=All_df.dropna(subset=['apache_4a_icu_death_prob'])

All_impute.drop(All_impute[All_impute['apache_4a_icu_death_prob']<0].index,inplace=True)#incorrect value

All_y_impute=All_impute['apache_4a_icu_death_prob']#get y

All_df_impute=All_impute.drop('apache_4a_icu_death_prob',axis=1)



data_impute=All_df_impute.copy()

y_impute = np.array(All_y_impute.tolist())

random_state = 23



X_train, X_test, y_train, y_test = train_test_split(data_impute, y_impute, test_size = 0.2, random_state = random_state)#stratify = y_impute)

X_train=pd.DataFrame(X_train,columns=data_impute.columns)

X_test=pd.DataFrame(X_test,columns=data_impute.columns)

lgbm_reg = lgbm.LGBMRegressor(n_estimators=300, random_state = 23,categorical_feature=category)

lgbm_reg.fit(X_train, y_train)

y_pred = lgbm_reg.predict(X_test)

acc_lgbm_reg = explained_variance_score(y_test,y_pred)

print('The accuracy of imputing apache_4a_icu_death_prob in all data is:  '+str(acc_lgbm_reg))



lgbm_reg.fit(data_impute,y_impute)

All_df_X=All_df.drop('apache_4a_icu_death_prob',axis=1)

All_df_icu=lgbm_reg.predict(All_df_X)



All_df_icu=np.reshape(All_df_icu,(131021,1))



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

All_df_icu=scaler.fit_transform(All_df_icu)



All_df.loc[All_df['apache_4a_icu_death_prob']<0,'apache_4a_icu_death_prob']=np.nan

All_df['apache_4a_icu_death_prob'].fillna(pd.Series(np.reshape(All_df_icu,(131021,))),inplace=True)



All_impute=All_df.dropna(subset=['apache_4a_hospital_death_prob'])

All_impute.drop(All_impute[All_impute['apache_4a_hospital_death_prob']<0].index,inplace=True)#incorrect value

All_y_impute=All_impute['apache_4a_hospital_death_prob']#get y

All_df_impute=All_impute.drop('apache_4a_hospital_death_prob',axis=1)



data_impute=All_df_impute.copy()

y_impute = np.array(All_y_impute.tolist())

random_state = 23



X_train, X_test, y_train, y_test = train_test_split(data_impute, y_impute, test_size = 0.2, random_state = random_state)#stratify = y_impute)

X_train=pd.DataFrame(X_train,columns=data_impute.columns)

X_test=pd.DataFrame(X_test,columns=data_impute.columns)

lgbm_reg = lgbm.LGBMRegressor(n_estimators=300, random_state = 23,categorical_feature=category)

lgbm_reg.fit(X_train, y_train)

y_pred = lgbm_reg.predict(X_test)

acc_lgbm_reg = explained_variance_score(y_test,y_pred)



print('The accuracy of imputing apache_4a_hospital_death_prob in all data is:  '+str(acc_lgbm_reg))

lgbm_reg.fit(data_impute,y_impute)

All_df_X=All_df.drop('apache_4a_hospital_death_prob',axis=1)

All_df_hos=lgbm_reg.predict(All_df_X)

All_df_hos=np.reshape(All_df_hos,(131021,1))

scaler = MinMaxScaler()

All_df_hos=scaler.fit_transform(All_df_hos)



All_df.loc[All_df['apache_4a_hospital_death_prob']<0,'apache_4a_hospital_death_prob']=np.nan

All_df['apache_4a_hospital_death_prob'].fillna(pd.Series(np.reshape(All_df_hos,(131021,))),inplace=True)
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
data=All_df[0:91713]

Test_data=All_df[91713:131021]

y = np.array(All_y[0:91713].tolist())

random_state = 23

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.2, random_state = random_state, stratify = y)

X_train=pd.DataFrame(X_train,columns=data.columns)

X_test=pd.DataFrame(X_test,columns=data.columns)
%%time

lgbm_clf = lgbm.LGBMClassifier(boosting_type='dart',n_estimators=1000,random_state = 23,categorical_feature=category,metric='auc')

lgbm_clf.fit(X_train, y_train)

y_pred = lgbm_clf.predict(X_test)

y_score = lgbm_clf.predict_proba(X_test)[:,1]
model_performance('lgbm_clf')
Best_params={'bagging_fraction': 0.8823860189717492, 'feature_fraction': 0.5012324406004431, 'lambda_l1': 2.6520965471163143, 'lambda_l2': 5.309685230258841, 'learning_rate': 0.09506145340497216, 'max_bin': 61, 'max_depth': 14, 'min_split_gain': 0.846642883905048, 'num_leaves':10}
lgbm_clf_rd = lgbm.LGBMClassifier(early_stopping_round=50,boosting_type='dart',objective='binary',random_state=23, silent=True, metric='auc', num_iterations=1000,categorical_feature=category,**Best_params,random_seed=23)
lgbm_clf_rd.fit(X_train, y_train)

y_pred = lgbm_clf_rd.predict(X_test)

y_score = lgbm_clf_rd.predict_proba(X_test)[:,1]
model_performance('lgbm_clf_rd')
#model output

lgbm_clf_rd = lgbm.LGBMClassifier(early_stopping_round=50,boosting_type='dart',objective='binary',num_iterations=1500,random_state=23, silent=True, metric='auc', categorical_feature=category,**Best_params,random_seed=23)

lgbm_clf_rd.fit(data, y)

y1_pred = lgbm_clf_rd.predict_proba(Test_data)

sub_df=pd.read_csv('/kaggle/input/widsdatathon2020/solution_template.csv')

output=pd.DataFrame()

output['encounter_id']=sub_df['encounter_id']

output['hospital_death']=y1_pred[:,1]

output.reset_index(drop=True,inplace=True)

output.to_csv('submission.csv',index=False)