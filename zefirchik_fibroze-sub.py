import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
SUB =  pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

MY_SUB = pd.read_csv("../input/fibrose-zefir/submission.csv")

TEST = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
PACIENT = TEST["Patient"].unique()

TRAIN_NEW = pd.DataFrame()

for ID in PACIENT:

    data = TEST[TEST.Patient == ID].copy()

    data.reset_index(inplace=True,drop=True)

    data = data[:1]

    r = range(-12,134)

    count_week = len(r)

    data = data.loc[data.index.repeat(count_week)].reset_index(drop=True)

    week_predict = [i for i in r]

    data["week_predict"] = week_predict

    TRAIN_NEW = pd.concat([TRAIN_NEW, data], ignore_index=True)

TEST = TRAIN_NEW

TEST['Patient_Week'] = TEST.agg('{0[Patient]}_{0[week_predict]}'.format, axis=1)
# SUB2 = pd.merge(TEST, MY_SUB, on='Patient_Week', how='left')

# SUB2[["FVC","Confidence"]] = SUB2[["FVC_y","Confidence"]]

# SUB2 = SUB2[["Patient_Week","FVC","Confidence"]]

# SUB2["Confidence"]=200

# SUB2.loc[SUB2["FVC"]<2000,["FVC"]]=2000
SUB2 = pd.merge(SUB, MY_SUB, on='Patient_Week', how='left')

SUB2[["FVC","Confidence"]] = SUB2[["FVC_y","Confidence_y"]]

SUB2 = SUB2[["Patient_Week","FVC","Confidence"]]

SUB2["Confidence"]=100

SUB2["FVC"]=2000
SUB2.to_csv("submission.csv", index=False)