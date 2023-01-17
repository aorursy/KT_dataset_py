# 사용법: 
# 1) 스크립트를 실행하기 전에 Ensemble 폴더를 먼저 만듭니다. 
# 2) 앙상블할 submission 화일을 Ensemble 폴더에 저장합니다.
# 3) 실행하면 현재 폴더에 앙상블한 submission 화일이 생성됩니다.
#
# 주) 이 스크립트는 Kaggle Kernel에서 실행할 수 없고 여러분의 Jupyter Notebook에서 실행해야 합니다.

import pandas as pd
import numpy as np
import os

folder = 'Ensemble'
nf = 0
for f in os.listdir(folder):
    ext = os.path.splitext(f)[-1]
    if ext == '.csv': 
        s = pd.read_csv(folder+"/"+f)
    else: 
        continue
    if len(s.columns) !=2:
        continue
    if nf == 0: 
        slist = s
    else: 
        slist = pd.merge(slist, s, on="custid")
    nf += 1

if nf >= 2:
    pred = 1
    for j in range(nf): pred = pred * slist.iloc[:,j+1] 
    pred = pred**(1/nf)

    submit = pd.DataFrame({'custid': slist.custid, 'gender': pred})
    t = pd.Timestamp.now()
    fname = "submission_ES_" + str(t.month) + str(t.day) + "_" + str(t.hour) + str(t.minute) + ".csv"
    submit.to_csv(fname, index=False)