import numpy as np

from sklearn.preprocessing import OneHotEncoder
def softmax(x, axis=1):

    # Stable Softmax

    # from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    y = x- np.max(x, axis=axis, keepdims=True)

    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)



def softmax_derivatives(s):

    s = s.reshape(-1,1)

    return np.diagflat(s) - np.dot(s, s.T)



def softmax_second_derivative(s, C=4):

    d2 = np.zeros((C, C))

    for i in range(C):

        for j in range(C):

            if i == j:

                d2[i, j] = (1. - s[i]) * s[i] * (1. - 2. * s[i])

            else:

                d2[i, j] = s[i] * s[j] * (2. * s[i] - 1.)

    return d2
def kappa_grads_lightgbm(y_true, y_pred):

    

    number_classes = 4

    C = number_classes

    labels = y_true 



    batch_size = y_true.shape[0]

    labels_one_hot = OneHotEncoder(categories=

                           [range(C)]*1, sparse=False).fit_transform(y_true.reshape(-1, 1))

    y_pred = np.reshape(y_pred, (batch_size, C), order='F')



    y_pred = softmax(y_pred)

    eps = 1e-12



    wtc = (y_true.reshape(-1, 1) - range(C))**2 / ((C - 1)**2)

    N = np.sum(wtc * y_pred)

    dN_ds = wtc



    Ni = np.sum(labels_one_hot, 0) / batch_size 

    repeat_op = np.tile(np.reshape(range(0, C), [C, 1]), [1, C])

    repeat_op_sq = np.square((repeat_op - np.transpose(repeat_op)))

    wij = repeat_op_sq / ((C - 1) ** 2)

    

    hist_preds = np.sum(y_pred, axis=0)

    D = np.sum(Ni.reshape(-1, 1) * (wij * hist_preds.reshape(1, -1)))

    dD_ds = np.tile(np.dot(wij, Ni), (batch_size, 1))

    

    dL_ds = dN_ds / (N + eps) - dD_ds / (D + eps)



    dL_da = np.zeros_like(dL_ds)

    ds_da = np.zeros((batch_size, C, C))

    for i in range(batch_size):

        ds_da[i] = softmax_derivatives(y_pred[i])

        dL_da[i] = np.dot(ds_da[i], dL_ds[i])



    d2L_da2 = np.zeros_like(dL_da)

    for b in range(batch_size):

        d2s_da2 = softmax_second_derivative(y_pred[b])

        d2N = -np.dot(dN_ds[b].reshape([C, 1]), dN_ds[b].reshape(1, C)) / (N * N + eps)

        d2D = np.dot(dD_ds[b].reshape([C, 1]), dD_ds[b].reshape(1, C)) / (D * D + eps)

        d2L_ds2 = d2N + d2D

        for c in range(C):

            AA = ds_da[b,0,c]*(ds_da[b,0,c] * d2L_ds2[0,0] + 

                                ds_da[b,1,c] * d2L_ds2[0,1] + 

                                ds_da[b,2,c] * d2L_ds2[0,2] +

                                ds_da[b,3,c] * d2L_ds2[0,3]

                                ) + dL_ds[b,0] * d2s_da2[c, 0] 



            BB = ds_da[b,1,c]*(ds_da[b,0,c] * d2L_ds2[1,0] + 

                                ds_da[b,1,c] * d2L_ds2[1,1] + 

                                ds_da[b,2,c] * d2L_ds2[1,2] +

                                ds_da[b,3,c] * d2L_ds2[1,3]

                                ) + dL_ds[b,1] * d2s_da2[c, 1] 



            CC = ds_da[b,2,c]*(ds_da[b,0,c] * d2L_ds2[2,0] + 

                                ds_da[b,1,c] * d2L_ds2[2,1] + 

                                ds_da[b,2,c] * d2L_ds2[2,2] +

                                ds_da[b,3,c] * d2L_ds2[2,3]

                                ) + dL_ds[b,2] * d2s_da2[c, 2] 



            DD = ds_da[b,3,c]*(ds_da[b,0,c] * d2L_ds2[3,0] + 

                                ds_da[b,1,c] * d2L_ds2[3,1] + 

                                ds_da[b,2,c] * d2L_ds2[3,2] +

                                ds_da[b,3,c] * d2L_ds2[3,3]

                                ) + dL_ds[b,3] * d2s_da2[c, 3] 



            d2L_da2[b, c] = AA + BB + CC + DD

    



    return [dL_da.flatten('F'), np.abs(d2L_da2.flatten('F'))]
!pip install autograd
import autograd.numpy as np

from autograd import grad

from autograd import jacobian

from autograd import hessian
# Test data

y_pred = np.array([[ 0.89912265,  0.79084255,  0.32162871, -0.99296229],

       [ 0.81017273,  0.18127493,  0.13865968, -0.32750946],

       [ 1.24011418,  0.94562047, -0.19091468, -0.68148713],

       [-1.10852297, -0.5044101 ,  0.41754522,  2.25403507],

       [ 1.41286477, -0.43752987, -0.34757177, -0.35673979],

       [ 3.19812033,  0.15338679, -1.12337413, -1.27692332],

       [ 2.4100999 ,  0.23849515, -0.79835829, -1.22977774],

       [-1.20118161, -1.21880044,  0.59375328,  3.13776406],

       [ 0.35147176,  0.03808217, -0.14641639, -0.22850701],

       [ 2.8068574 ,  0.30871835, -0.84488727, -1.28116301]])



y_true = np.array([3, 0, 3, 3, 0, 0, 0, 3, 3, 0])
def weighted_kappa(y_true, y_pred, C=4):

    

    batch_size = y_true.shape[0]

    labels_one_hot = OneHotEncoder(categories=

                           [range(C)]*1, sparse=False).fit_transform(y_true.reshape(-1, 1))

    y_pred = softmax(y_pred)

    eps = 1e-12



    wtc = (y_true.reshape(-1, 1) - range(C))**2 / ((C - 1)**2)

    N = np.sum(wtc * y_pred)

    Ni = np.sum(labels_one_hot, 0) / batch_size 



    repeat_op = np.tile(np.reshape(range(0, C), [C, 1]), [1, C])

    repeat_op_sq = np.square((repeat_op - np.transpose(repeat_op)))

    wij = repeat_op_sq / ((C - 1) ** 2)



    histp = np.sum(y_pred, axis=0)

    D = np.sum(Ni.reshape(-1, 1) * (wij * histp.reshape(1, -1)))

    

    return np.log(N / (D + eps))
kappa_grad = grad(weighted_kappa, 1)

kappa_hess = hessian(weighted_kappa, 1)
[custom_grad, custom_hess] = kappa_grads_lightgbm(y_true, y_pred)

autograd_grad = kappa_grad(y_true, y_pred).flatten('F')
np.allclose(custom_grad, autograd_grad)
autograd_all_hess = kappa_hess(y_true, y_pred)

# just need the diagonals to compare

autograd_diag_hess = np.array([

    autograd_all_hess[i, j, i, j]

    for j in range(autograd_all_hess.shape[1])

    for i in range(autograd_all_hess.shape[0])

    ])
# use absolute val of autograd_hess since we used it in the custom loss

np.allclose(custom_hess, np.abs(autograd_diag_hess))