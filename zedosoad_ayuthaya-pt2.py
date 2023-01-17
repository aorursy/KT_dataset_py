!pip install pydensecrf
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from spectral import *

import pickle

from math import sqrt, pi, exp, floor, ceil

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

import pydensecrf.densecrf as dcrf

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, unary_from_softmax, create_pairwise_gaussian



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
### INFO

# img 1: 180

# img 2: 304

# img 3: 180
crop_size = 112

crop_rate = 0.75
predictions_test_dict = np.load("../input/pred3/predictions_3.npy").item()
### layer 0: source; layer 1: transport



# abrir imagens

imgs = []

for i in range(3):

    img = envi.open('../input/lesgoooo/stack_prim_source_region{}.hdr'.format(i+1))

    imgs.append(np.array(img.open_memmap(writeable = True)))



    

# criar layer transport; por source e transport no fim 

for im in imgs:

    im[:,:,0] /= 2

    im[:,:,1] = np.clip(im[:,:,1] - im[:,:,0], 0, 1)

    tmp = im[:,:,0:2].copy()

    im[:,:,0:5] = im[:,:,2::]

    im[:,:,5::] = tmp.copy()



for i in range(3):

    imgs[i] = np.concatenate((imgs[i], 1 - np.clip(imgs[i][:,:,-2] + imgs[i][:,:,-1], 0, 1)[:,:,None]), axis = 2)
def full_img(idx_test):



    window_size = crop_size

    rate = crop_rate

    image_size_y = test_img.shape[0]

    image_size_x = test_img.shape[1]

    

    Y_test = test_img[:,:,5::]

    X_test = test_img[:,:,0:5]





    def gaussian_dist_to_center(window_size):

        half_window = window_size//2

        gauss_dists = np.zeros((window_size, window_size))

        mat_aux = np.zeros((half_window, half_window))



        center = half_window - 1

        std = half_window/2





        for i in range(half_window):

            for j in range(half_window):

                d = sqrt((i - center)**2 + (j - center)**2)

                mat_aux[i,j] = 1/(std*sqrt(2*pi))*exp(-0.5*(d/std)**2)



        gauss_dists[0:(center+1),0:(center+1)] = mat_aux

        gauss_dists[0:(center+1),(center+1):window_size] = mat_aux[:,::-1]

        gauss_dists[(center+1):window_size,0:(center+1)] = mat_aux[::-1,:]

        gauss_dists[(center+1):window_size,(center+1):window_size] = mat_aux[::-1,::-1]

        return gauss_dists





    def get_overlap_weights2(mat, overlap_pix):

        #right, left

        weights = np.zeros(overlap_pix)



        for i in range(overlap_pix):

            xd = mat.shape[0] - overlap_pix + i

            weights[i] = mat[0,i]/(mat[0,xd] + mat[0,i])



        return weights





    def get_overlap_weights4(mat, overlap_pix_y, overlap_pix_x):

        #down-right, down-left, up-right, up-left

        weights = np.zeros((overlap_pix_y, overlap_pix_x, 4))



        for i in range(overlap_pix_y):

            for j in range(overlap_pix_x):

                xd = mat.shape[0] - overlap_pix_x + j

                yb = mat.shape[0] - overlap_pix_y + i

                weights[i,j,0] = mat[yb,xd]/(mat[i,xd] + mat[yb,xd] + mat[i,j] + mat[yb,j])

                weights[i,j,1] = mat[yb,j]/(mat[i,xd] + mat[yb,xd] + mat[i,j] + mat[yb,j])

                weights[i,j,2] = mat[i,xd]/(mat[i,xd] + mat[yb,xd] + mat[i,j] + mat[yb,j])

                weights[i,j,3] = mat[i,j]/(mat[i,xd] + mat[yb,xd] + mat[i,j] + mat[yb,j])



        return weights





    gauss_dists = gaussian_dist_to_center(window_size)









    weights = get_overlap_weights2(gauss_dists, round((1-rate)*window_size))



    w_regular_e = weights

    w_regular_d = weights[::-1]





    #canto

    weightsc = get_overlap_weights4(gauss_dists, round((1-rate)*window_size), round((1-rate)*window_size))



    w_canto_eb = weightsc[:,:,0]

    w_canto_db = weightsc[:,:,1]

    w_canto_ec = weightsc[:,:,2]

    w_canto_dc = weightsc[:,:,3]







    # last right

    aux = (image_size_x - round((1-rate)*window_size))/(window_size*rate)

    aux = aux - floor(aux)

    if aux == 0:

        pix_x = round((1-rate)*window_size)

    else:

        pix_x = window_size - round(aux*window_size*rate)



    weightslr = get_overlap_weights2(gauss_dists, pix_x)



    w_last_cols_e = weightslr

    w_last_cols_d = weightslr[::-1]







    #canto right

    weights = get_overlap_weights4(gauss_dists, round((1-rate)*window_size), pix_x)



    w_cantor_eb = weights[:,:,0]

    w_cantor_db = weights[:,:,1]

    w_cantor_ec = weights[:,:,2]

    w_cantor_dc = weights[:,:,3]









    # last down

    aux = (image_size_y - round((1-rate)*window_size))/(window_size*rate)

    aux = aux - floor(aux)

    if aux == 0:

        pix_y = round((1-rate)*window_size)

    else:

        pix_y = window_size - round(aux*window_size*rate)



    weightsld = get_overlap_weights2(gauss_dists, pix_y)



    w_last_rows_e = weightsld

    w_last_rows_d = weightsld[::-1]







    #canto down

    weights = get_overlap_weights4(gauss_dists, pix_y, round((1-rate)*window_size))



    w_cantod_eb = weights[:,:,0]

    w_cantod_db = weights[:,:,1]

    w_cantod_ec = weights[:,:,2]

    w_cantod_dc = weights[:,:,3]





    #canto last

    weights = get_overlap_weights4(gauss_dists, pix_y, pix_x)



    w_canto_last_eb = weights[:,:,0]

    w_canto_last_db = weights[:,:,1]

    w_canto_last_ec = weights[:,:,2]

    w_canto_last_dc = weights[:,:,3]











    n_classes = 3

    # stitch

    preds = predictions_test

    pred_im = np.zeros((image_size_y, image_size_x, n_classes))



    movex = round(window_size*rate)

    movey = movex

    correcty = window_size-movey

    correctx = window_size-movex

    n_ims_y = ceil((image_size_y-correcty)/(window_size*rate))

    n_ims_x = ceil((image_size_x-correctx)/(window_size*rate))





    y = 0

    for i in range(n_ims_y-2):

        if i == 0:

            y = 0

            correcty = 0

        else:

            if i == 1:

                y = y + window_size

            else:

                y = y + movey

            correcty = window_size-movey



        for j in range(n_ims_x-2):



            if j == 0:

                x = 0

                correctx = 0

            else:

                if j == 1:

                    x = x + window_size

                else:

                    x = x + movex

                correctx = window_size-movex

            



            pred_im[y:(y+movey-correcty), x:(x+movex-correctx), :] = preds[i*n_ims_x + j, correcty:movey, correctx:movex, :]

            pred_im[y:(y+movey-correcty), (x+movex-correctx):(x+window_size-correctx), :] = (preds[i*n_ims_x + j, correcty:movey, movex::, :] * np.repeat(np.tile(w_regular_d, (movey-correcty, 1))[:,:,np.newaxis], n_classes, axis = 2)

                + preds[i*n_ims_x + j + 1, correcty:movey, 0:(window_size-movex), :] * np.repeat(np.tile(w_regular_e, (movey-correcty, 1))[:,:,np.newaxis], n_classes, axis = 2))

            pred_im[(y+movey-correcty):(y+window_size-correcty), x:(x+movex-correctx), :] = (preds[i*n_ims_x + j, movey::, correctx:movex, :] * np.repeat(np.tile(w_regular_d, (movex-correctx, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

                + preds[(i+1)*n_ims_x + j, 0:(window_size-movey), correctx:movex, :] * np.repeat(np.tile(w_regular_e, (movex-correctx, 1)).T[:,:,np.newaxis], n_classes, axis = 2))

            pred_im[(y+movey-correcty):(y+window_size-correcty), (x+movex-correctx):(x+window_size-correctx), :] = (preds[i*n_ims_x + j, movey::, movex::, :] * np.repeat(w_canto_eb[:,:,np.newaxis], n_classes, axis = 2)

                + preds[i*n_ims_x + j + 1, movey::, 0:(window_size-movex), :] * np.repeat(w_canto_db[:,:,np.newaxis], n_classes, axis = 2)

                + preds[(i + 1)*n_ims_x + j, 0:(window_size-movey), movex::, :] * np.repeat(w_canto_ec[:,:,np.newaxis], n_classes, axis = 2)

                + preds[(i + 1)*n_ims_x + j + 1, 0:(window_size-movey), 0:(window_size-movex), :] * np.repeat(w_canto_dc[:,:,np.newaxis], n_classes, axis = 2))



        x = x + movex

        j = j + 1

        pred_im[y:(y+movey-correcty), x:(x+movex-pix_x), :] = preds[i*n_ims_x + j, correcty:movey, correctx:-pix_x, :]



        aux1 = (preds[i*n_ims_x + j, correcty:movey, -pix_x::, :] * np.repeat(np.tile(w_last_cols_d, (movey-correcty, 1))[:,:,np.newaxis], n_classes, axis = 2)

             + preds[i*n_ims_x + j + 1, correcty:movey, 0:pix_x, :] * np.repeat(np.tile(w_last_cols_e, (movey-correcty, 1))[:,:,np.newaxis], n_classes, axis = 2))



        aux3 = (preds[i*n_ims_x + j, movey::, -pix_x::, :] * np.repeat(w_cantor_eb[:,:,np.newaxis], n_classes, axis = 2)

            + preds[i*n_ims_x + j + 1, movey::, 0:pix_x, :] * np.repeat(w_cantor_db[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j, 0:(window_size-movey), -pix_x::, :] * np.repeat(w_cantor_ec[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j + 1, 0:(window_size-movey), 0:pix_x, :] * np.repeat(w_cantor_dc[:,:,np.newaxis], n_classes, axis = 2))





        if x >= x+movex-pix_x:

            pred_im[y:(y+movey-correcty), x:(x+movex), :] = aux1[:, (pix_x - movex)::, :]

            pred_im[(y+movey-correcty):(y+window_size-correcty), x:(x+movex), :] = aux3[:, (pix_x - movex)::, :]



        else:

            pred_im[y:(y+movey-correcty), (x+movex-pix_x):(x+movex), :] = aux1



            aux2 = (preds[i*n_ims_x + j, movey::, correctx:-pix_x, :] * np.repeat(np.tile(w_regular_d, (movey-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

                + preds[(i + 1)*n_ims_x + j, 0:(window_size-movey), correctx:-pix_x, :] * np.repeat(np.tile(w_regular_e, (movey-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2))



            pred_im[(y+movey-correcty):(y+window_size-correcty), x:(x+movex-pix_x), :] = aux2



            pred_im[(y+movey-correcty):(y+window_size-correcty), (x+movex-pix_x):(x+movex), :] = aux3



        x = x + movex

        j = j + 1



        pred_im[y:(y+movey-correcty), x::, :] = preds[i*n_ims_x + j, correcty:movey, pix_x::, :]

        aux2 = (preds[i*n_ims_x + j, movey::, pix_x::, :] * np.repeat(np.tile(w_regular_d, (window_size-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j, 0:(window_size-movey), pix_x::, :] * np.repeat(np.tile(w_regular_e, (window_size-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2))



        pred_im[(y+movey-correcty):(y+window_size-correcty), x::, :] = aux2



    if i == 0:

        y = y + window_size

    else:

        y = y + movey

    i = i + 1

    correcty = window_size-movey



    for j in range(n_ims_x-2):

        if j == 0:

            x = 0

            correctx = 0

        else:

            if j == 1:

                x = x + window_size

            else:

                x = x + movex

            correctx = window_size-movex





        pred_im[y:(y+movey-pix_y), x:(x+movex-correctx), :] = preds[i*n_ims_x + j, correcty:-pix_y, correctx:movex, :]













        aux2 = pred_im[(y+movey-pix_y):(y+window_size-correcty), x:(x+movex-correctx), :] = (preds[i*n_ims_x + j, -pix_y::, correctx:movex, :] * np.repeat(np.tile(w_last_rows_d, (movex-correctx, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i+1)*n_ims_x + j, 0:pix_y, correctx:movex, :] * np.repeat(np.tile(w_last_rows_e, (movex-correctx, 1)).T[:,:,np.newaxis], n_classes, axis = 2))



        aux3 = (preds[i*n_ims_x + j, -pix_y::, movex::, :] * np.repeat(w_cantod_eb[:,:,np.newaxis], n_classes, axis = 2)

            + preds[i*n_ims_x + j + 1, -pix_y::, 0:(window_size-movex), :] * np.repeat(w_cantod_db[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j, 0:pix_y, movex::, :] * np.repeat(w_cantod_ec[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j + 1, 0:pix_y, 0:(window_size-movex), :] * np.repeat(w_cantod_dc[:,:,np.newaxis], n_classes, axis = 2))





        if y >= y+movey-pix_y:

            pred_im[y:(y+movey), x:(x+movex-correctx), :] = aux2[(pix_y - movey)::, :, :]

            pred_im[y:(y+movey), (x+movex-correctx):(x+window_size-correctx), :] = aux3[(pix_y - movey)::, :, :]



        else:

            pred_im[(y+movey-pix_y):(y+movey), x:(x+movex-correctx), :] = aux2



            aux1 = (preds[i*n_ims_x + j, correcty:-pix_y, movex::, :] * np.repeat(np.tile(w_regular_d, (movey-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

                + preds[i*n_ims_x + j + 1, correcty:-pix_y, 0:(window_size-movex), :] * np.repeat(np.tile(w_regular_e, (movey-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))



            pred_im[y:(y+movey-pix_y), (x+movex-correctx):(x+window_size-correctx), :] = aux1



            pred_im[(y+movey-pix_y):(y+movey), (x+movex-correctx):(x+window_size-correctx), :] = aux3







    x = x + movex

    j = j + 1





    pred_im[y:(y+movey-pix_y), x:(x+movex-pix_x), :] = preds[i*n_ims_x + j, correcty:-pix_y, correctx:-pix_x, :]





    aux3 = (preds[i*n_ims_x + j, -pix_y::, -pix_x::, :] * np.repeat(w_canto_last_eb[:,:,np.newaxis], n_classes, axis = 2)

        + preds[i*n_ims_x + j + 1, -pix_y::, 0:pix_x, :] * np.repeat(w_canto_last_db[:,:,np.newaxis], n_classes, axis = 2)

        + preds[(i + 1)*n_ims_x + j, 0:pix_y, -pix_x::, :] * np.repeat(w_canto_last_ec[:,:,np.newaxis], n_classes, axis = 2)

        + preds[(i + 1)*n_ims_x + j + 1, 0:pix_y, 0:pix_x, :] * np.repeat(w_canto_last_dc[:,:,np.newaxis], n_classes, axis = 2))





    if x >= x+movex-pix_x and y >= y+movey-pix_y:

        pred_im[y:(y+window_size-correcty), x:(x+window_size-correctx), :] = aux3[(pix_y - movey)::, (pix_x - movex)::, :]

    elif y >= y+movey-pix_y:

        aux2 = (preds[i*n_ims_x + j, -pix_y::, correctx:-pix_x, :] * np.repeat(np.tile(w_last_rows_d, (movex-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j, 0:pix_y, correctx:-pix_x, :] * np.repeat(np.tile(w_last_rows_e, (movex-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2))

        pred_im[y:(y+window_size-correcty), x:(x+movex-pix_x), :] = aux2[(pix_y - movey)::, :, :]

        pred_im[y:(y+window_size-correcty), (x+movex-pix_x):(x+window_size-correctx), :] = aux3[(pix_y - movey)::, :, :]



    elif x >= x+movex-pix_x:

        aux1 = (preds[i*n_ims_x + j, correcty:-pix_y, -pix_x::, :] * np.repeat(np.tile(w_last_cols_d, (movex-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

                + preds[i*n_ims_x + j + 1, correcty:-pix_y, 0:pix_x, :] * np.repeat(np.tile(w_last_cols_e, (movey-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))



        pred_im[y:(y+movey-pix_y), x:(x+movex), :] = aux1[:, (pix_x - movex)::, :]

        pred_im[(y+movey-pix_y):(y+window_size-correcty), x:(x+window_size-correctx), :] = aux3[:, (pix_x - movex)::, :]



    else:

        aux1 = (preds[i*n_ims_x + j, correcty:-pix_y, -pix_x::, :] * np.repeat(np.tile(w_last_cols_d, (movey-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

            + preds[i*n_ims_x + j + 1, correcty:-pix_y, 0:pix_x, :] * np.repeat(np.tile(w_last_cols_e, (movey-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))



        pred_im[y:(y+movey-pix_y), (x+movex-pix_x):(x+movex), :] = aux1



        aux2 = (preds[i*n_ims_x + j, -pix_y::, correctx:-pix_x, :] * np.repeat(np.tile(w_last_rows_d, (movex-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

            + preds[(i + 1)*n_ims_x + j, 0:pix_y, correctx:-pix_x, :] * np.repeat(np.tile(w_last_rows_e, (movex-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2))



        pred_im[(y+movey-pix_y):(y+window_size-correcty), x:(x+movex-pix_x), :] = aux2

        pred_im[(y+movey-pix_y):(y+window_size-correcty), (x+movex-pix_x):(x+window_size-correctx), :] = aux3





    x = x + movex

    j = j + 1



    pred_im[y:(y+movey-pix_y), x::, :] = preds[i*n_ims_x + j, correcty:-pix_y, pix_x::, :]

    aux2 = (preds[i*n_ims_x + j, -pix_y::, pix_x::, :] * np.repeat(np.tile(w_last_rows_d, (window_size-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2)

        + preds[(i + 1)*n_ims_x + j, 0:pix_y, pix_x::, :] * np.repeat(np.tile(w_last_rows_e, (window_size-pix_x, 1)).T[:,:,np.newaxis], n_classes, axis = 2))



    if y >= y+movey-pix_y:

        pred_im[y:(y+window_size-correcty), x::, :] = aux2[(pix_y - movey)::, :, :]

    else:

        pred_im[(y+movey-pix_y):(y+window_size-correcty), x::, :] = aux2







    y = y + movey

    #y = pred_im.shape[0] - window_size + pix_y

    i = i + 1



    for j in range(n_ims_x-2):

        if j == 0:

            x = 0

            correctx = 0

        else:

            if j == 1:

                x = x + window_size

            else:

                x = x + movex

            correctx = window_size-movex



        pred_im[y::, x:(x+movex-correctx), :] = preds[i*n_ims_x + j, pix_y::, correctx:movex, :]



        pred_im[y::, (x+movex-correctx):(x+window_size-correctx), :] = (preds[i*n_ims_x + j, pix_y::, movex::, :] * np.repeat(np.tile(w_regular_d, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

                + preds[i*n_ims_x + j + 1, pix_y::, 0:(window_size-movex), :] * np.repeat(np.tile(w_regular_e, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))



    x = x + movex

    j = j + 1



    if x >= x+movex-pix_x:

        aux1 = (preds[i*n_ims_x + j, pix_y::, -pix_x:, :] * np.repeat(np.tile(w_last_cols_d, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

            + preds[i*n_ims_x + j + 1, pix_y::, 0:pix_x, :] * np.repeat(np.tile(w_last_cols_e, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))

        pred_im[y::, x:(x+movex), :] = aux1[:, (pix_x - movex)::, :]

    else:

        pred_im[y::, x:(x+movex-pix_x), :] = preds[i*n_ims_x + j, pix_y::, correctx:-pix_x, :]



        pred_im[y::, (x+movex-pix_x):(x+movex), :] = (preds[i*n_ims_x + j, pix_y::, -pix_x::, :] * np.repeat(np.tile(w_last_cols_d, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2)

            + preds[i*n_ims_x + j + 1, pix_y::, 0:pix_x, :] * np.repeat(np.tile(w_last_cols_e, (window_size-pix_y, 1))[:,:,np.newaxis], n_classes, axis = 2))



    x = x + movex

    j = j + 1

    pred_im[y::, x::, :] = preds[i*n_ims_x + j, pix_y::, pix_x::, :]



    ytrue = np.argmax(Y_test[:,:,[2,0,1]], axis = 2)

    ypred = pred_im[:,:,[2,0,1]]



    ypred_discrete = np.argmax(ypred, axis = 2)

    



    image_compare = np.zeros((ytrue.shape[0], ytrue.shape[1]*2 + 5))

    image_compare[:, 0:ytrue.shape[1]] = ypred_discrete

    image_compare[:, ytrue.shape[1]:(ytrue.shape[1]+5)] = np.ones((ytrue.shape[0],5))

    image_compare[:, (ytrue.shape[1]+5)::] = ytrue

    



    plt.figure(figsize=(1/100 * image_compare.shape[1], 1/100 * image_compare.shape[0]))

    plt.imshow(image_compare, cmap="gray")

    plt.show()



    np.save("pred.npy", ypred_discrete)





    conf = confusion_matrix(ytrue.flatten(), ypred_discrete.flatten())

    print("confusion matrix")

    print(conf)

    UI0 = conf[0,0]/(conf[1,0] + conf[0,0] + conf[0,1] + conf[0,2] + conf[2,0])

    UI1 = conf[1,1]/(conf[1,0] + conf[1,1] + conf[1,2] + conf[0,1] + conf[2,1])

    UI2 = conf[2,2]/(conf[2,0] + conf[2,1] + conf[2,2] + conf[0,2] + conf[1,2])

    UIM = (UI0 + UI1 + UI2)/3

    UIM2 = (UI1 + UI2)/2

    P0prod = (conf[0,0])/(conf[0,0] + conf[0,1] + conf[0,2])

    P1prod = (conf[1,1])/(conf[1,0] + conf[1,1] + conf[1,2])

    P2prod = (conf[2,2])/(conf[2,0] + conf[2,1] + conf[2,2])

    P0user = (conf[0,0])/(conf[0,0] + conf[1,0] + conf[2,0])

    P1user = (conf[1,1])/(conf[0,1] + conf[1,1] + conf[2,1])

    P2user = (conf[2,2])/(conf[0,2] + conf[1,2] + conf[2,2])

    ACC = (conf[0,0] + conf[1,1] + conf[2,2])/(np.sum(conf))

    

    evals[num_idx]['normal'] = {}

    evals[num_idx]['normal']['ui0'] = UI0

    evals[num_idx]['normal']['ui1'] = UI1

    evals[num_idx]['normal']['ui2'] = UI2

    evals[num_idx]['normal']['uim'] = UIM

    evals[num_idx]['normal']['uim2'] = UIM2

    evals[num_idx]['normal']['p0prod'] = P0prod

    evals[num_idx]['normal']['p1prod'] = P1prod

    evals[num_idx]['normal']['p2prod'] = P2prod

    evals[num_idx]['normal']['p0user'] = P0user

    evals[num_idx]['normal']['p1user'] = P1user

    evals[num_idx]['normal']['p2user'] = P2user

    evals[num_idx]['normal']['acc'] = ACC

    evals[num_idx]['normal']['confusion'] = conf





    print("IoU bg: {:.2f}%, IoU source: {:.2f}%, IoU transport: {:.2f}%\nIoU Mean: {:.2f}%, IoU Mean2: {:.2f}%".format(UI0*100, UI1*100, UI2*100, UIM*100, UIM2*100))

    print("Prod")

    print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0prod*100, P1prod*100, P2prod*100))

    print("User")

    print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0user*100, P1user*100, P2user*100))

    print("overall acc: {:.2f}%".format(ACC*100))





    crfs = [8]

    

    for lol in range(1):

        # CRF

        pred_after_CRF = np.ndarray((ytrue.shape[0], ytrue.shape[1]))

        probs = np.zeros((ytrue.shape[0], ytrue.shape[1], 3))

        

        image = X_test.transpose((2, 0, 1))

        probs = ypred



        unary = unary_from_softmax(probs.transpose((2, 0, 1)).reshape((3,-1)))



        # The inputs should be C-continious -- we are using Cython wrapper

        unary = np.ascontiguousarray(unary)





        d = dcrf.DenseCRF(image.shape[1] * image.shape[2], 3)



        d.setUnaryEnergy(unary)

        # This potential penalizes small pieces of segmentation that are

        # spatially isolated -- enforces more spatially consistent segmentations

        #feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[1:3])



        #d.addPairwiseEnergy(feats, compat=3,

        #                    kernel=dcrf.DIAG_KERNEL,

        #                    normalization=dcrf.NORMALIZE_SYMMETRIC)





        # This creates the color-dependent features --

        # because the segmentation that we get from CNN are too coarse

        # and we can use local color features to refine them

        

        feats = create_pairwise_bilateral(sdims=(5, 5), schan=(0.01,), img=image, chdim=0)

        #feats = create_pairwise_bilateral(sdims=(3, 3), schan=(10, 10, 10),

        #                                   img=image, chdim=0)



        d.addPairwiseEnergy(feats, compat=10)









        Q = d.inference(crfs[lol])

        pred_after_CRF[:,:] = np.argmax(Q, axis = 0).reshape((image.shape[1], image.shape[2]))



        if idx_test == 0:

            #plt.figure(figsize=(1/100 * image_compare.shape[1], 1/100 * image_compare.shape[0]))

            plt.imshow(pred_after_CRF[-(20+727):-20,:], cmap="gray")

            plt.show()

        elif idx_test == 1:

            #plt.figure(figsize=(1/100 * image_compare.shape[1], 1/100 * image_compare.shape[0]))

            plt.imshow(pred_after_CRF[:800,:800], cmap="gray")

            plt.show()

        else:

            #plt.figure(figsize=(1/100 * image_compare.shape[1], 1/100 * image_compare.shape[0]))

            plt.imshow(pred_after_CRF[:500,:500], cmap="gray")

            plt.show()





        image_compare = np.zeros((ytrue.shape[0], ytrue.shape[1]*2 + 5))

        image_compare[:, 0:ytrue.shape[1]] = pred_after_CRF

        image_compare[:, ytrue.shape[1]:(ytrue.shape[1]+5)] = np.ones((ytrue.shape[0],5))

        image_compare[:, (ytrue.shape[1]+5)::] = ytrue



        plt.figure(figsize=(1/100 * image_compare.shape[1], 1/100 * image_compare.shape[0]))

        plt.imshow(image_compare, cmap="gray")

        plt.show()

        

        np.save("CRF{}.npy".format(lol), pred_after_CRF)



        conf = confusion_matrix(ytrue.flatten(), pred_after_CRF.flatten())

        print("confusion matrix")

        print(conf)

        UI0 = conf[0,0]/(conf[1,0] + conf[0,0] + conf[0,1] + conf[0,2] + conf[2,0])

        UI1 = conf[1,1]/(conf[1,0] + conf[1,1] + conf[1,2] + conf[0,1] + conf[2,1])

        UI2 = conf[2,2]/(conf[2,0] + conf[2,1] + conf[2,2] + conf[0,2] + conf[1,2])

        UIM = (UI0 + UI1 + UI2)/3

        UIM2 = (UI1 + UI2)/2

        P0prod = (conf[0,0])/(conf[0,0] + conf[0,1] + conf[0,2])

        P1prod = (conf[1,1])/(conf[1,0] + conf[1,1] + conf[1,2])

        P2prod = (conf[2,2])/(conf[2,0] + conf[2,1] + conf[2,2])

        P0user = (conf[0,0])/(conf[0,0] + conf[1,0] + conf[2,0])

        P1user = (conf[1,1])/(conf[0,1] + conf[1,1] + conf[2,1])

        P2user = (conf[2,2])/(conf[0,2] + conf[1,2] + conf[2,2])

        ACC = (conf[0,0] + conf[1,1] + conf[2,2])/(np.sum(conf))

        

        evals[num_idx]['crf{}'.format(lol)] = {}

        evals[num_idx]['crf{}'.format(lol)]['ui0'] = UI0

        evals[num_idx]['crf{}'.format(lol)]['ui1'] = UI1

        evals[num_idx]['crf{}'.format(lol)]['ui2'] = UI2

        evals[num_idx]['crf{}'.format(lol)]['uim'] = UIM

        evals[num_idx]['crf{}'.format(lol)]['uim2'] = UIM2

        evals[num_idx]['crf{}'.format(lol)]['p0prod'] = P0prod

        evals[num_idx]['crf{}'.format(lol)]['p1prod'] = P1prod

        evals[num_idx]['crf{}'.format(lol)]['p2prod'] = P2prod

        evals[num_idx]['crf{}'.format(lol)]['p0user'] = P0user

        evals[num_idx]['crf{}'.format(lol)]['p1user'] = P1user

        evals[num_idx]['crf{}'.format(lol)]['p2user'] = P2user

        evals[num_idx]['crf{}'.format(lol)]['acc'] = ACC

        evals[num_idx]['crf{}'.format(lol)]['confusion'] = conf

        



        print("IoU bg: {:.2f}%, IoU source: {:.2f}%, IoU transport: {:.2f}%\nIoU Mean: {:.2f}%, IoU Mean2: {:.2f}%".format(UI0*100, UI1*100, UI2*100, UIM*100, UIM2*100))

        print("Prod")

        print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0prod*100, P1prod*100, P2prod*100))

        print("User")

        print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0user*100, P1user*100, P2user*100))

        print("overall acc: {:.2f}%".format(ACC*100))

        





    return(UIM)
evals = dict()

num_idx = 0

#big_ensemble = []



for i in range(3):

    test_img = imgs[i]

    for j in range(5):

        predictions_test = predictions_test_dict[5*i + j]

        evals[num_idx] = {}

        full_img(i)

        num_idx += 1

    aux = list()

    for j in range(5):

        aux.append(predictions_test_dict[i*5 + j])

    

    prediction = np.stack(aux, axis=0)

    predictions_test = np.mean(prediction, axis = 0)

    evals[num_idx] = {}

    full_img(i)

    num_idx += 1
def print_conf(conf):

    UI0 = conf[0,0]/(conf[1,0] + conf[0,0] + conf[0,1] + conf[0,2] + conf[2,0])

    UI1 = conf[1,1]/(conf[1,0] + conf[1,1] + conf[1,2] + conf[0,1] + conf[2,1])

    UI2 = conf[2,2]/(conf[2,0] + conf[2,1] + conf[2,2] + conf[0,2] + conf[1,2])

    UIM = (UI0 + UI1 + UI2)/3

    UIM2 = (UI1 + UI2)/2

    P0prod = (conf[0,0])/(conf[0,0] + conf[0,1] + conf[0,2])

    P1prod = (conf[1,1])/(conf[1,0] + conf[1,1] + conf[1,2])

    P2prod = (conf[2,2])/(conf[2,0] + conf[2,1] + conf[2,2])

    P0user = (conf[0,0])/(conf[0,0] + conf[1,0] + conf[2,0])

    P1user = (conf[1,1])/(conf[0,1] + conf[1,1] + conf[2,1])

    P2user = (conf[2,2])/(conf[0,2] + conf[1,2] + conf[2,2])

    ACC = (conf[0,0] + conf[1,1] + conf[2,2])/(np.sum(conf))



    print(conf)

    print("IoU bg: {:.2f}%, IoU source: {:.2f}%, IoU transport: {:.2f}%\nIoU Mean: {:.2f}%, IoU Mean2: {:.2f}%".format(UI0*100, UI1*100, UI2*100, UIM*100, UIM2*100))

    print("Prod")

    print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0prod*100, P1prod*100, P2prod*100))

    print("User")

    print("Acc bg: {:.2f}%, Acc source: {:.2f}%, Acc transport: {:.2f}%".format(P0user*100, P1user*100, P2user*100))

    print("overall acc: {:.2f}%".format(ACC*100))
# results



print("---BEST---")

print("normal\n")

# best normal

confusion_matrix = np.zeros((3,3))



aux = 0

for _ in range(3):

    max_aux = 0

    idx_aux = 0

    for i in range(aux, aux + 5):

        if evals[i]['normal']['uim'] > max_aux:

            max_aux = evals[i]['normal']['uim']

            idx_aux = i        

    confusion_matrix += evals[idx_aux]['normal']['confusion']

    aux += 6

print_conf(confusion_matrix)



print("\nCRF\n")

# best crf

confusion_matrix = np.zeros((3,3))



aux = 0

for _ in range(3):

    max_aux = 0

    idx_aux = 0

    for i in range(aux, aux + 5):

        if evals[i]['crf0']['uim'] > max_aux:

            max_aux = evals[i]['crf0']['uim']

            idx_aux = i        

    confusion_matrix += evals[idx_aux]['crf0']['confusion']

    aux += 6

print_conf(confusion_matrix)



print("\n\n---WORST---")

print("normal\n")

# worst normal

confusion_matrix = np.zeros((3,3))



aux = 0

for _ in range(3):

    min_aux = 100

    idx_aux = 0

    for i in range(aux, aux + 5):

        if evals[i]['normal']['uim'] < min_aux:

            min_aux = evals[i]['normal']['uim']

            idx_aux = i        

    confusion_matrix += evals[idx_aux]['normal']['confusion']

    aux += 6

print_conf(confusion_matrix)



# worst crf

print("\nCRF\n")

confusion_matrix = np.zeros((3,3))



aux = 0

for _ in range(3):

    min_aux = 100

    idx_aux = 0

    for i in range(aux, aux + 5):

        if evals[i]['crf0']['uim'] < min_aux:

            min_aux = evals[i]['crf0']['uim']

            idx_aux = i        

    confusion_matrix += evals[idx_aux]['crf0']['confusion']

    aux += 6

print_conf(confusion_matrix)





print("\n\n---MEAN---")

print("normal\n")

# mean normal

confusion_matrix = np.zeros((3,3))



for n in range(3):      

    confusion_matrix += evals[n*6 + 5]['normal']['confusion']

print_conf(confusion_matrix)



print("\nCRF\n")

# mean crf

confusion_matrix = np.zeros((3,3))



for n in range(3):      

    confusion_matrix += evals[n*6 + 5]['crf0']['confusion']

print_conf(confusion_matrix)