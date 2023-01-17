import pandas as pd

import numpy as np

from sklearn.metrics import f1_score, accuracy_score

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/intercampusai2019/train.csv')

test = pd.read_csv('/kaggle/input/intercampusai2019/test.csv')

sample_submission2 = pd.read_csv('/kaggle/input/intercampusai2019/sample_submission2.csv')
test.shape, train.shape
print ('Ratio of Test to Train ',(16496 /(train.shape[0] +test.shape[0]), 38312 /(train.shape[0] +test.shape[0])))
def encode(df):

    df['Gender'] = df.Gender.replace({

        'Female': 0,

        'Male': 1,

        })

    df['Foreign_schooled'] = df.Foreign_schooled.replace({

    'No': 0,

    'Yes': 1,

    })

    df['Past_Disciplinary_Action'] = df.Past_Disciplinary_Action.replace({

    'No': 0,

    'Yes': 1,

    })

    df['Previous_IntraDepartmental_Movement'] = df.Previous_IntraDepartmental_Movement.replace({

    'No': 0,

    'Yes': 1,

    })

    df['No_of_previous_employers'] = df.No_of_previous_employers.replace({

    '0': 0,

    '1': 1,

    '2': 2,

    '3': 3,

    '4': 4,

    '5': 5,

    'More than 5': 6

    })

    df['Division'] = df.Division.replace({

    'Commercial Sales and Marketing': 0,

    'Customer Support and Field Operations': 1,

    'Sourcing and Purchasing': 2,

    'Information Technology and Solution Support': 3,

    'Information and Strategy': 4,

    'Business Finance Operations': 5,

    'People/HR Management': 6,

    'Regulatory and Legal services': 7,

    'Research and Innovation': 8,

    })

    df['Qualification'] = df.Qualification.replace({

    'First Degree or HND': 1,

    'MSc, MBA and PhD': 2,

    'Non-University Education': 0

    })

    df['Channel_of_Recruitment'] = df.Channel_of_Recruitment.replace({

    'Agency and others': 1,

    'Direct Internal process': 0,

    'Referral and Special candidates': 2

    })

    df['Marital_Status'] = df.Marital_Status.replace({

    'Married': 2,

    'Single': 1,

    'Not_Sure': 0

    })

    df['Last_performance_score'] = df.Last_performance_score.replace({

     0.0: 0,

     2.5: 1,

     5.0: 2,

     7.5: 3,

     10.0: 4,

     12.5: 5

    })

    
encode(train)

encode(test)
test.info()
all_data = pd.concat([train, test])

all_data.Qualification.fillna(0, inplace = True)

all_data['emplo_year'] = all_data.Year_of_recruitment - all_data.Year_of_birth

all_data['age'] = 2019 - all_data.Year_of_birth
# all_data['Training_score_average'][all_data['Training_score_average'] < 34]=34

# all_data['Training_score_average'][all_data['Training_score_average'] > 86]=86 

# # lgb best
all_data['score_2'] = (all_data.Training_score_average)**2

all_data['score_3'] = (all_data.Training_score_average)**3

all_data['score_perf'] = (all_data.Training_score_average)/(all_data.Last_performance_score)

all_data['score_perf2'] = (all_data.Training_score_average)+(all_data.Last_performance_score)

all_data['score_perf4'] = ((all_data.Training_score_average)+(all_data.Targets_met))/all_data.Trainings_Attended



all_data['score_perf3'] = ((all_data.Training_score_average)+(all_data.Last_performance_score))/all_data.Trainings_Attended

all_data['Targets_met_Previos'] = (all_data.Targets_met) + (all_data.Previous_Award)
all_data.columns
all_data['Division_Score'] = all_data['Division'] + np.digitize(all_data['Training_score_average'], [30, 40, 50, 60, 70, 80, 90, 100])

all_data['Targets_met_Score'] = all_data['Targets_met'] + np.digitize(all_data['Training_score_average'], [30, 40, 50, 60, 70, 80, 90, 100])

all_data['LPS_Score'] = all_data['Last_performance_score'] + np.digitize(all_data['Training_score_average'], [30, 40, 50, 60, 70, 80, 90, 100])

all_data['Division_Targets_met'] = all_data['Division'] + all_data['Targets_met']

all_data['Division_LPS'] = all_data['Last_performance_score'] + all_data['Targets_met']
all_data['max_training_by_Division'] = all_data['Division'].map(all_data.groupby('Division')['Training_score_average'].max())

all_data['Division_training_max_ratio'] = all_data['Training_score_average'] / all_data['max_training_by_Division']

all_data['min_training_by_Division'] = all_data['Division'].map(all_data.groupby('Division')['Training_score_average'].min())

all_data['Division_training_min_ratio'] = all_data['Training_score_average'] / all_data['min_training_by_Division']

all_data['std_training_by_Division'] = all_data['Division'].map(all_data.groupby('Division')['Training_score_average'].std())

all_data['Division_training_std_ratio'] = all_data['Training_score_average'] / all_data['std_training_by_Division']
all_data['mean_training_by_Division'] = all_data['Division'].map(all_data.groupby('Division')['Training_score_average'].mean())

all_data['Division_training_mean_Divider'] = all_data['Training_score_average'] / all_data['mean_training_by_Division']



all_data['mean_rating_by_Division'] = all_data['Division'].map(all_data.groupby(['Division','State_Of_Origin'])['Training_score_average'].mean())

all_data['Division_rating_mean_Divider'] = all_data['Training_score_average'] / all_data['mean_rating_by_Division']

all_data['mean_Targets_met_by_Division'] = all_data['Division'].map(all_data.groupby('Division')['Targets_met'].mean())

all_data.head()
all_data.columns
feat = ['Channel_of_Recruitment', 'Division', 'EmployeeNo',

        'Last_performance_score', 

       'Previous_Award',

       'Promoted_or_Not', 'Qualification','Targets_met',

       'Training_score_average', 'Trainings_Attended',

       'emplo_year', 'age','score_perf3',

       'score_perf', 'score_perf2', 'score_perf4', 

        'Division_Score','LPS_Score', 'Division_LPS',

       'Targets_met_Score', 'Division_Targets_met', 

       'mean_training_by_Division',

       'Division_training_mean_Divider', 'mean_Targets_met_by_Division','mean_rating_by_Division', 'Division_rating_mean_Divider']
# Label encode region

all_data.State_Of_Origin = all_data.State_Of_Origin.astype('category').cat.codes

all_data[['Qualification','Last_performance_score']] = all_data[['Qualification','Last_performance_score']].astype('int32')
train.shape
new_train = all_data[feat][:38312]

new_test = all_data[feat][38312:]



# new_train = all_data[:38312]

# new_test = all_data[38312:]
Target_name="Promoted_or_Not"

not_used_cols=[Target_name, 'EmployeeNo'

       ]

features_name=[ f for f in new_train.columns if f not in not_used_cols]
new_train.head()
params = {"objective": "binary",

          "booster": "gbtree",

          "eta": 0.01,

#           "max_depth":6,

#           'min_child_weight':2,

#           "categorical_feature":cat_feat,

#           'is_unbalance': True

          "subsample": 0.9,

#           "colsample_bytree": 0.7,

#           "gamma": 0.5,

#           "alpha": 0.04,

#           "min_child_weight": 10,

          'colsample_bytree': 0.8,     

          'eval_metric':'binary_logloss',

          'metric':'binary_logloss'

          }
model = []

import numpy as np, pandas as pd, lightgbm as lgb, warnings

from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

val_score =[]

train_score = []

best_boost = []



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1996)

oof =  np.zeros(len(new_train))

predictions = np.zeros(len(new_test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(new_train.values, new_train.Promoted_or_Not.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(new_train.iloc[trn_idx][features_name], label=new_train['Promoted_or_Not'].iloc[trn_idx])

    val_data = lgb.Dataset(new_train.iloc[val_idx][features_name], label=new_train['Promoted_or_Not'].iloc[val_idx],reference=trn_data)

    

    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=50, early_stopping_rounds = 100)

    model.append(clf)

    

    oof[val_idx] = clf.predict(new_train.iloc[val_idx][features_name], num_iteration=clf.best_iteration)

    predictions += clf.predict(new_test[features_name], num_iteration=clf.best_iteration) / folds.n_splits

    

    

    val_score.append(clf.best_score['valid_1']['binary_logloss'])

    train_score.append(clf.best_score['training']['binary_logloss'])

    best_boost.append(clf.best_iteration)
i=1

for mod in model:

    lgb.plot_importance(mod,xlabel='Feature importance' + str(i)) 

    i=i+1
print('Xgb cv_mean', np.asarray(val_score, dtype=np.float32).mean())

print('Xgb cv_std', np.asarray(val_score, dtype=np.float32).std())
print("micro F1 CV score: {:<8.5f}".format(f1_score(new_train['Promoted_or_Not']  , (oof>= 0.5).astype(int), average='micro')))  # present Best
print("F1 CV score: {:<8.5f}".format(f1_score(new_train['Promoted_or_Not'] , (oof>= 0.5).astype(int))))  # present Best
pd.DataFrame({"EmployeeNo": new_test.EmployeeNo.values, 'Promoted_or_Not':(predictions >= 0.5).astype(int)}).to_csv('output_submission.csv', index=False)