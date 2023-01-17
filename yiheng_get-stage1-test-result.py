from sklearn.metrics import log_loss

def calc_logloss(targets, outputs, eps=1e-6):

    logloss_classes = [log_loss(np.floor(targets[:,i]), np.clip(outputs[:,i], eps, 1-eps)) for i in range(6)]

    return np.average(logloss_classes, weights=[2,1,1,1,1,1])



import pandas as pd

import pickle

import os

import numpy as np

import warnings

warnings.filterwarnings("ignore")

dup = pd.read_csv("../input/stage1-test-gt/dup_s1_test.csv")

test = pd.read_csv("../input/stage1-test-gt/s1_test_results.csv")

test = test.merge(dup, on = 'SOPInstanceUID', how = 'left')
def get_split_result(filename, test, eps, rm_dup=False):

    f1 = pd.read_csv(filename)



    f1['type'] = f1['ID'].apply(lambda x: x.split('_')[2])

    f1['name'] = f1['ID'].apply(lambda x: x.split('_')[1])



    name = f1[['name']]



    f1_epidural = f1[['name','Label']][f1['type'] == 'epidural']

    f1_epidural.columns = ['name','epidural']

    f1_intraparenchymal = f1[['name','Label']][f1['type'] == 'intraparenchymal']

    f1_intraparenchymal.columns = ['name','intraparenchymal']

    f1_intraventricular = f1[['name','Label']][f1['type'] == 'intraventricular']

    f1_intraventricular.columns = ['name','intraventricular']

    f1_subarachnoid = f1[['name','Label']][f1['type'] == 'subarachnoid']

    f1_subarachnoid.columns = ['name','subarachnoid']

    f1_subdural = f1[['name','Label']][f1['type'] == 'subdural']

    f1_subdural.columns = ['name','subdural']

    f1_any = f1[['name','Label']][f1['type'] == 'any']

    f1_any.columns = ['name','any']



    name = name.merge(f1_any, on = 'name', how = 'left')

    name = name.merge(f1_epidural, on = 'name', how = 'left')

    name = name.merge(f1_intraparenchymal, on = 'name', how = 'left')

    name = name.merge(f1_intraventricular, on = 'name', how = 'left')

    name = name.merge(f1_subarachnoid, on = 'name', how = 'left')

    name = name.merge(f1_subdural, on = 'name', how = 'left')

    name = name.drop_duplicates()

    name.rename(columns = {'name': 'SOPInstanceUID'}, inplace=True)

    name['SOPInstanceUID'] = 'ID_' + name['SOPInstanceUID']

    

    name = name.merge(test, on = 'SOPInstanceUID', how = 'left')

    

    if rm_dup:

        name_use = name[name['dup'].isnull() == True] #remove duplicate patientID

    else:

        name_use = name.copy()  #all test

    gt = name_use[['any_y',

           'epidural_y', 'subdural_y', 'subarachnoid_y', 'intraventricular_y',

           'intraparenchymal_y']].values

    pred = name_use[['any',

               'epidural', 'subdural', 'subarachnoid', 'intraventricular',

               'intraparenchymal']].values

    return calc_logloss(gt, pred, eps=eps)
#come from https://www.kaggle.com/krishnakatyal/keras-efficientnet-b3

get_split_result("../input/kernel-0076/submission.csv", test, 1e-6)
get_split_result("../input/kernel-0076/submission.csv", test, 1e-6, rm_dup=True)