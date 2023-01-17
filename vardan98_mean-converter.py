import pandas as pd
import os
folders = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B7']
base = '../input/siimisc/kfold'
df = pd.read_csv('../input/siimisc/kfold/B0/sub_EfficientNetB0_224 (1).csv')
submits_mean = [df.copy(), df.copy(), df.copy(), df.copy(), df.copy(), df.copy(), df.copy()]
submits_maj = [df.copy(), df.copy(), df.copy(), df.copy(), df.copy(), df.copy(), df.copy()]
for i, folder in enumerate(folders):
    base2 = os.path.join(base, folder)
    files = os.listdir(base2)
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(os.path.join(base2, file))['target'])
    for j in range(10982):
        mn = 0
        maj = 0
        t = 0
        for k in range(5):
            mn += dfs[k][j]
            if dfs[k][j] > 0.5:
                t += 1
        mn = mn/5
        if t > 2:
            for k in range(5):
                if dfs[k][j] > 0.5:
                    maj += dfs[k][j]
            maj = maj/t
        else:
            for k in range(5):
                if dfs[k][j] < 0.5:
                    maj += dfs[k][j]
            maj = maj/(5-t)
            
        submits_mean[i].iloc[j, df.columns.get_loc('target')] = mn
        submits_maj[i].iloc[j, df.columns.get_loc('target')] = maj
        
for i, submit in enumerate(submits_mean):
    out_path = folders[i] + '-mean.csv'
    submit.to_csv(out_path, index=False)

for i, submit in enumerate(submits_maj):
    out_path = folders[i] + '-maj.csv'
    submit.to_csv(out_path, index=False)
folders = ['B6']
submits_mean = [df.copy()]
submits_maj = [df.copy()]
for i, folder in enumerate(folders):
    base2 = os.path.join(base, folder)
    files = os.listdir(base2)
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(os.path.join(base2, file))['target'])
    for j in range(10982):
        mn = 0
        maj = 0
        t = 0
        for k in range(3):
            mn += dfs[k][j]
            if dfs[k][j] > 0.5:
                t += 1
        mn = mn/3
        if t > 1:
            for k in range(3):
                if dfs[k][j] > 0.5:
                    maj += dfs[k][j]
            maj = maj/t
        else:
            for k in range(3):
                if dfs[k][j] < 0.5:
                    maj += dfs[k][j]
            maj = maj/(3-t)
            
        submits_mean[i].iloc[j, df.columns.get_loc('target')] = mn
        submits_maj[i].iloc[j, df.columns.get_loc('target')] = maj
        
for i, submit in enumerate(submits_mean):
    out_path = folders[i] + '-mean.csv'
    submit.to_csv(out_path, index=False)

for i, submit in enumerate(submits_maj):
    out_path = folders[i] + '-maj.csv'
    submit.to_csv(out_path, index=False)
base = '../input/siimisc/majority_mean/majority_mean'
df = pd.read_csv('../input/siimisc/kfold/B0/sub_EfficientNetB0_224 (1).csv')
submits_mean = [df.copy()]
submits_maj = [df.copy()]
files = os.listdir(base)
dfs = []
for file in files:
    dfs.append(pd.read_csv(os.path.join(base, file))['target'])
for j in range(10982):
    mn = 0
    maj = 0
    t = 0
    for k in range(8):
        mn += dfs[k][j]
        if dfs[k][j] > 0.5:
            t += 1
    mn = mn/8
    if t > 4:
        for k in range(8):
            if dfs[k][j] > 0.5:
                maj += dfs[k][j]
        maj = maj/t
    elif t < 3:
        for k in range(8):
            if dfs[k][j] < 0.5:
                maj += dfs[k][j]
        maj = maj/(8-t)
    else:
        maj = mn
    submits_mean[0].iloc[j, df.columns.get_loc('target')] = mn
    submits_maj[0].iloc[j, df.columns.get_loc('target')] = maj

for i, submit in enumerate(submits_mean):
    out_path = 'majmean-mean.csv'
    submit.to_csv(out_path, index=False)

for i, submit in enumerate(submits_maj):
    out_path = 'majmean-majmean.csv'
    submit.to_csv(out_path, index=False)