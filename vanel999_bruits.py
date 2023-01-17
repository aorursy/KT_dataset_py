# Ajout de bruit dans les données

import glob

import pandas as pd

import numpy as np

from numpy.linalg import eig

from numpy.linalg import norm

from random import *

from scipy.spatial.distance import euclidean

#from fastdtw import fastdtw

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing





def genereBruitNormal(sigma, mu, nbPoints):

    data = np.random.randn(nbPoints) * sigma + mu

    return data



def genereBruitExponentiel(p, nbPoints):

    data = []

    for i in range(nbPoints):

        data.append(expovariate(p))

    return np.array(data)



def genereBruitUniform(p, nbPoints):

    data = np.random.uniform(0, p, nbPoints)

    return data



def noisyDataSet(df, p, mu = 0, loi = 'normal'):

    nbLign, nbCol = df.shape

    nbPoints = nbCol - 1

    df_result = df.copy()

    for i in range(nbLign):

        bruit = pd.DataFrame()

        if loi == 'normal':

            bruit = genereBruitNormal(p, mu, nbPoints)

        elif loi == 'uniforme':

            bruit = genereBruitUniform(p, nbPoints)

        else:

            bruit = genereBruitExponentiel(p, nbPoints)

            

        cls = df_result.iloc[:, 0]

        df_result = df_result.drop(0, axis=1)

        df_result = df_result + bruit

        df_result.insert(0, 0, cls)

        

    return df_result



def noisyDataSets(lst_df, pmin, pmax, pas, lst_output_file, mu = 0, loi = 'normal'):

    n = len(lst_df)

    for i in range(n):

        d = lst_df[i]

        _, nbPoints = d.shape

        for p in np.arange(pmin,pmax, pas):

            df_r = noisyDataSet(d, p, loi = loi )

            name = "{}_{}_{}.csv".format(lst_output_file[i], p, loi)

            print(name)

            df_r.to_csv(name, header = False, index = False)

    return 0
#Main, ajout de bruits aux donnees

df_training_set = pd.read_csv('../input/coffee/Coffee_TRAIN.tsv',delimiter='\t',encoding='utf-8', header = None, index_col = None)

df_testing_set = pd.read_csv('../input/coffee/Coffee_TEST.tsv',delimiter='\t',encoding='utf-8', header = None, index_col = None)



df_cofee = pd.concat([df_training_set, df_testing_set])





for l in ['uniforme', 'normal', 'exponentiel']:

    noisyDataSets([df_cofee], 3, 10, 1,  lst_output_file = ["cofee"],  loi = l)
def covariance(x, y, w):

    npx = np.array(x)

    npy = np.array(y)

    

    nx = len(npx)

    ny = len(npy)

    

    if nx != ny:

        print("Attention, les deux vecteurs doivent avoir la même longue. Longueur(x) = {}, Longueur(y) = {}".format(nx, ny))

        return -1

    

    if w > nx:

        print("Attention, la longueur de la fenêtre doit être inférieure à celle du vecteur, w = {}, n = {}".format(w, nx))

        return -1

    

    tau = np.zeros((w, w))

    idf = (w - 1)

    tau = np.outer(npx[(idf- w + 1):idf], npy[(idf - w + 1):idf])

    

    for idf in range(w, nx):

        aux = np.outer(npx[(idf- w + 1):idf], npy[(idf - w + 1):idf])

        tau += aux

    

    return tau



def auto_covariance(x, w):

    return covariance(x, x, w)



def U(X, k = 2):

    eigVal, eigVect = eig(X)

    idx = eigVal.argsort()[-k:][::-1]

    eigVal = eigVal[idx]

    eigVect = eigVect[:, idx]

    

    return eigVal, eigVect



def F(X, Y):

    return norm( (X-Y), ord = 'fro')



def ED(x, y):

    npx = np.array(x)

    npy = np.array(y)

    return norm((npx-npy))



def DTW(x, y):

    npx = np.array(x)

    npy = np.array(y)

    

    distance, path = fastdtw(npx, npy, dist=euclidean)

    

    return distance



def matriceDistED(df):

    nbL, nbC = df.shape

    D = np.zeros((nbL, nbL))

    for i in range(nbL):

        for j in range(i, nbL):

            D[i, j] = ED(np.array(df.iloc[i, :]), np.array(df.iloc[j, :]))

            D[j, i] = D[i, j]

    return D



def matriceDistDTW(df):

    nbL, nbC = df.shape

    D = np.zeros((nbL, nbL))

    for i in range(nbL):

        for j in range(i, nbL):

            D[i, j] = DTW(df.iloc[i, :], df.iloc[j, :])

            D[j, i] = D[i, j]

    return D



def matriceDistF(df):

    nbL, nbC = df.shape

    D = np.zeros((nbL, nbL))

    

    lst_U = []

    

    for k in range(nbL):

        Ax = auto_covariance(df.iloc[k, :], nbC - 3)

        _, Ux = U(Ax)

        lst_U.append(Ux)

        

        

    for i in range(nbL):

        for j in range(i, nbL):

            Ux = lst_U[i]

            Uy = lst_U[j]

            

            D[i, j] = F(Ux, Uy)

            D[j, i] = D[i, j]

            

    return D



def matriceDistF_s(df):

    nbL, nbC = df.shape

    D = np.zeros((nbL, nbL))

    

    lst_U = []

    lst_A = []

    

    for k in range(nbL):

        Ax = auto_covariance(df.iloc[k, :], nbC - 3)

        lst_A.append(Ax)

        _, Ux = U(Ax)

        lst_U.append(Ux)

    

    for i in range(nbL):

        for j in range(i, nbL):

            Ax = lst_A[i]

            Ay = lst_A[j]

            

            Ux = lst_U[i]

            Uy = lst_U[j]

            

            D[i, j] = F(Ax.dot(Ux), Ay.dot(Uy))

            D[j, i] = D[i, j]

    return D



def idxPPV(v):

    d = pd.DataFrame(v)

    d.sort_values(0, inplace=True)

    smallest = d.reset_index().iloc[0]['index']

    return int(smallest)



def perf(ids_ts_trng, ids_ts_tstg, cls_ts, M_dist):

    j = 0

    

    cls_ts_trng = cls_ts[ np.array(ids_ts_trng) ]

    cls_ts_tstg = cls_ts[ np.array(ids_ts_tstg) ]

    

    cls_cal_ts_trng = []

    

    

    for idx in ids_ts_trng:

        dists = M_dist[np.array(ids_ts_tstg), idx]

        i = idxPPV(dists)

        cls_cal_ts_trng.append( cls_ts_tstg[i])

        j = j + 1

    cfs_mtr = confusion_matrix(cls_ts_trng, cls_cal_ts_trng)

    n,_ = cfs_mtr.shape

    tp = 0

    lst_f1 = []

    lst_recall = []

    lst_precision = []

    for k in range(n):

        tp += cfs_mtr[k,k]

    for k in range(n):

        fp = np.sum(cfs_mtr[:, k])-cfs_mtr[k, k]

        fn = np.sum(cfs_mtr[k, :])-cfs_mtr[k, k]

        

        p, r = 0,0

        f1 = 0

        if tp != 0:

            p = tp / (tp + fp)

            r = tp / (tp + fn)

        if p != 0 and r != 0:

            f1 = 2 * (p * r) / (p + r)

        

        lst_f1.append(f1)

        lst_precision.append(p)

        lst_recall.append(r)

        

    return  [np.mean(lst_precision), np.mean(lst_recall), np.mean(lst_f1)]



def KFCV(path, k):

    

    lst_jeuxDeDonnees = []

    lst_lois = []

    lst_std = []

    lst_dist = []

    lst_p = []

    lst_r = []

    lst_f1 = []

    

    lst_path_filenames = glob.glob(path + "/*.csv")

    cv = KFold(n_splits=k, random_state=42, shuffle=False)

    

    for path_filename in lst_path_filenames:

        

        df = pd.read_csv(path_filename, header = None, index_col = None)

        cls = df.iloc[:, 0]

        df = df.drop(0, axis=1)

        print(path_filename)

        filename = path_filename.split("/")[-1]

        aux = filename.split("_")

        

        dataset_name = aux[0]

    

        std = round( float(aux[1]),1)      

        loi = aux[2].split(".")[0]

    

        for d in ['ED', 'FOTS']:#, 'FOTS*','DTW']

            

            print(d)

            M = np.array([])

            if d == 'ED':

                M = matriceDistED(df)

            elif d == 'DTW':

                M = matriceDistDTW(df)

            elif d == 'FOTS':

                M = matriceDistF(df)

            else:

                M = matriceDistF_s(df)

            

            scores = []

                

            for train_index, test_index in cv.split(df):

                idx_trng = np.array(train_index)

                idx_tstg = np.array(test_index)

                scores.append( perf(idx_trng, idx_tstg, np.array(cls), M) )

                

            df_scores = pd.DataFrame(scores)

            score_mean = df_scores.mean(axis = 0)

            

            lst_jeuxDeDonnees.append(dataset_name)

            lst_std.append(std)

            lst_lois.append(loi)

            lst_dist.append(d)

            lst_p.append(score_mean[0])

            lst_r.append(score_mean[1])

            lst_f1.append(score_mean[2])

    dic = {"jeuxDeDonnees":lst_jeuxDeDonnees, "std":lst_std, "lois":lst_lois, "dist":lst_dist, "precision":lst_p, "rappel":lst_r, "f1":lst_f1}

    d_r = pd.DataFrame(dic)

    return d_r
path = "."

r = KFCV(path, 5)

r.groupby(["std", "lois", "dist"]).mean()