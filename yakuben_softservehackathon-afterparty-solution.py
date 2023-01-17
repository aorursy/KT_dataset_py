import numpy as np

import pandas as pd



from datetime import datetime, date, time, timedelta



from tqdm import tqdm, trange
def convert2datetime(text):

    if not text:

        return np.nan

    month, day, year = [int(el) for el in text.split('/')]

    if year < 30:

        year = 2000+year

    else:

        year = 1900+year

    

    return date(year, month, day)



def timedelta2month(timedelta):

    return round(timedelta.days/30.5)
employees = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/employees.csv', converters={'HiringDate':convert2datetime,'DismissalDate':convert2datetime})

history = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/history.csv', converters={'Date':convert2datetime})

submission = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/submission.csv')



category_cols = ['PositionID', 'CustomerID', 'ProjectID', 'DevCenterID', 'SBUID', 'FunctionalOfficeID', 'CompetenceGroupID', 'PaymentTypeId',]

boolean_cols = ['IsTrainee', 'IsInternalProject', 'OnSite']

continious_cols = ['Utilization', 'WageGross', 'BonusOneTime']

ordinal_cols = ['PositionLevel', 'LanguageLevelID', 'HourVacation', 'HourMobileReserve', 'HourLockedReserve', 'MonthOnPosition', 'MonthOnSalary', 'APM']

other_cols = ['EmployeeID', 'Date']
dismissal_date = history['EmployeeID'].map(employees.set_index('EmployeeID')['DismissalDate'])

days2dismissal = (dismissal_date - history['Date'])

days2dismissal = days2dismissal.fillna(timedelta(days=9999))

month2dismissal = days2dismissal.apply(timedelta2month)

history['target'] = (month2dismissal <= 3).astype(int)

history.loc[history['Date'] > date(2018, 11, 1), 'target'] = None
history['ProjectID'] = history['ProjectID'].fillna('other')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

history['CustomerID'] = le.fit_transform(history['CustomerID'])

history['ProjectID'] = le.fit_transform(history['ProjectID'])
employees_dismissal_date = employees[employees['DismissalDate'].notna()].set_index('EmployeeID')['DismissalDate']

bad_history_mask = history['Date'] >= history['EmployeeID'].map(employees_dismissal_date)

history = history[~bad_history_mask].reset_index(drop=True)
employees_start = employees.set_index('EmployeeID')['HiringDate']

history_employees_start = history['EmployeeID'].map(employees_start)

history['company_work_time'] = (history['Date'] - history_employees_start).map(timedelta2month)

history.loc[history['company_work_time'] >= 12*10, 'company_work_time'] = 12*10



history = history[history['company_work_time'] >= 4].reset_index()
history.groupby('Date')['MonthOnSalary'].mean().plot(figsize=(10, 5));
%%time

old_employees = employees.loc[employees['HiringDate'] < date(2017, 7, 1), 'EmployeeID']

for emp in old_employees:

    emp_history = history[history['EmployeeID'] == emp]

    

    old_value = 0

    for el in emp_history['MonthOnSalary']:

        if el > old_value:

            old_value = el

        else:

            break

          

    moth_from_salary = max(0, 12-old_value)

    

    history.loc[emp_history.index[:old_value], 'MonthOnSalary'] += moth_from_salary

    

history.loc[history['MonthOnSalary'] > 13, 'MonthOnSalary'] = 14
history.groupby('Date')['MonthOnSalary'].mean().plot(figsize=(10, 5));
history.groupby('Date')['MonthOnPosition'].mean().plot(figsize=(10, 5));
%%time



time_from_hiring = (date(2017, 7, 1)-employees.set_index('EmployeeID')['HiringDate']).map(timedelta2month).map(lambda x: max(0,x))

time_from_hiring /= 2.5

time_from_hiring = time_from_hiring.round().astype(int)



old_employees = employees.loc[employees['HiringDate'] < date(2017, 7, 1), 'EmployeeID']

for emp in old_employees:

    emp_history = history[history['EmployeeID'] == emp]



    old_value = 0

    for el in emp_history['MonthOnPosition']:

        if el > old_value:

            old_value = el

        else:

            break

            

    history.loc[emp_history.index[:old_value], 'MonthOnPosition'] += time_from_hiring[emp]



history.loc[history['MonthOnPosition'] > 5*12, 'MonthOnPosition'] = 5*12
history.groupby('Date')['MonthOnPosition'].mean().plot(figsize=(10, 5));
def normalize(col):

    return (col - col.mean())/col.std()



norm_cols = ['WageGross', 'BonusOneTime', 'HourVacation', 'HourMobileReserve', 'HourLockedReserve']#, 'APM', 'Utilization', 'MonthOnPosition', 'MonthOnSalary']

for month in history['Date'].unique():

    date_idx = history.index[history['Date'] == month]

    for col in norm_cols:

        history.loc[date_idx, col] = normalize(history.loc[date_idx, col])
test_date = date(2019, 2, 1)

train_date = [date(2018, 11, 1), date(2018, 8, 1)]



test = history[(history['Date'] == test_date) & history['EmployeeID'].isin(submission['EmployeeID'])]

test = test.sort_values('EmployeeID').reset_index(drop=True)



train = history[(history['Date'] <= train_date[0])&(history['Date'] >= train_date[1])].reset_index(drop=True)
cv_dates = [(date(2018, 8, 1), date(2018, 5, 1), date(2018, 11, 1)), 

            (date(2018, 7, 1), date(2018, 4, 1), date(2018, 10, 1)),

            (date(2018, 6, 1), date(2018, 3, 1), date(2018, 9, 1)),

            (date(2018, 5, 1), date(2018, 2, 1), date(2018, 8, 1)),

            (date(2018, 4, 1), date(2018, 1, 1), date(2018, 7, 1)),

            (date(2018, 3, 1), date(2017, 12, 1), date(2018, 6, 1)),

           ]



cv_data = []

for cv_date  in cv_dates:

    fold_train = history[(history['Date'] <= cv_date[0])&(history['Date'] >= cv_date[1])].reset_index(drop=True)

    fold_val = history[history['Date'] == cv_date[2]].reset_index(drop=True)

    cv_data.append((fold_train, fold_val))
train_drop_cols = ['target','Date','EmployeeID']#, 'CustomerID', 'ProjectID', 'DevCenterID', 'SBUID', 'PositionID']
X_test = test.drop(columns=train_drop_cols)



X_train = train.drop(columns=train_drop_cols)

y_train = train['target']
cv_X_y = []

for fold_train, fold_val  in cv_data:

    fold_train_X = fold_train.drop(columns=train_drop_cols)

    fold_train_y = fold_train['target']

    

    fold_val_X = fold_val.drop(columns=train_drop_cols)

    fold_val_y = fold_val['target']

    

    cv_X_y.append(((fold_train_X, fold_train_y), (fold_val_X, fold_val_y)))
from sklearn.model_selection import train_test_split

from sklearn.metrics import fbeta_score





def f1_7(estimator, X, y):

    return fbeta_score(y, (estimator.predict_proba(X)[:,1] > 0.5).astype(int), 1.7)





def calk_best_score(true, pred):

    max_score = 0

    best_tresh = 0

    for tresh in np.arange(0.01, 1, 0.01):

        pred_0_1 = pred>tresh

        if any(pred_0_1):

            score = fbeta_score(true, pred_0_1, 1.7, labels=[0,1])

            if score > max_score:

                max_score = score

                best_tresh = tresh



    return max_score, best_tresh



def validate_model(model, cv_X_y, n=5):

    best_score_mass = []

    tresh_mass = []

    predict_proba_mass = []

    

    for (fold_train_X, fold_train_y), (fold_val_X, fold_val_y) in cv_X_y:

        fold_scores = []

        fold_tresh = []

        fold_proba = []

        for i in range(n):

            model.fit(fold_train_X, fold_train_y)

            predict_proba = model.predict_proba(fold_val_X)[:,1]

            max_score, best_tresh = calk_best_score(fold_val_y, predict_proba)

            

            fold_scores.append(max_score)

            fold_tresh.append(best_tresh)

            fold_proba.append(predict_proba)



        best_score_mass.append(fold_scores)

        tresh_mass.append(fold_tresh)

        predict_proba_mass.append(fold_proba)



    tresh_mass = np.array(tresh_mass)

    best_score_mass = np.array(best_score_mass)

    



    mean_tresh = tresh_mass.mean()

    

    score_mass = []

    for fold_proba, (_, (_, fold_val_y)) in zip(predict_proba_mass, cv_X_y):

        fold_score_mass = []

        for proba in fold_proba:

            score = fbeta_score(fold_val_y, proba>mean_tresh, 1.7, labels=[0,1])

            fold_score_mass.append(score)

        score_mass.append(fold_score_mass)

    score_mass = np.array(score_mass)

    

    return mean_tresh, score_mass, best_score_mass, tresh_mass

    

    

def validate_result_vizualize(mean_tresh, score_mass, best_score_mass, tresh_mass):

    print('Best score by fold:')

    print(best_score_mass)

    print('Score by sample:')

    print(score_mass)

    print('Tresh by sample:')

    print(tresh_mass)

    print()

    print('Best score by fold:')

    print(best_score_mass.mean(axis=1))

    print('Score by fold:')

    print(score_mass.mean(axis=1))

    print('Tresh by fold:')

    print(tresh_mass.mean(axis=1))

    print()

    print('Best score std:')

    print(best_score_mass.mean(axis=1).std())

    print('Score std:')

    print(score_mass.mean(axis=1).std())

    print('Tresh std:')

    print(tresh_mass.mean(axis=1).std())

    print()

    print(f'Best score mean: {best_score_mass.mean()}')

    print(f'Score mean     : {score_mass.mean()}')

    print(f'Tresh mean: {mean_tresh}')



    

def create_sub(name, model, submission, X_train, y_train, X_test, tresh):

    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:,1]

    test_prediction = (pred_proba > tresh).astype(int)

    submission['target'] = test_prediction

    submission.to_csv(name, index=False)

    return submission
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=1000, 

                             class_weight='balanced',

                             max_leaf_nodes=64,

                             max_depth=12,

                             max_features=5, 

                             min_samples_split=5, 

                             min_samples_leaf=3,

                             criterion='gini',#'entropy', 

                             n_jobs=-1)



mean_tresh, score_mass, best_score_mass, tresh_mass = validate_model(model, cv_X_y, n=5)

validate_result_vizualize(mean_tresh, score_mass, best_score_mass, tresh_mass)
model.n_estimators = 30000
sub = create_sub('submission.csv', model, submission, X_train, y_train, X_test, mean_tresh)

print(sub['target'].mean())

sub