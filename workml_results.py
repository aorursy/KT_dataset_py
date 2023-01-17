# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fnl_acc=np.load('/kaggle/input/densenet-last-fold/final_accuracy_last.npy',allow_pickle=True).item()

bst_acc=np.load('/kaggle/input/densenet-last-fold/best_accuracy_last.npy',allow_pickle=True).item()

hist=np.load('/kaggle/input/densenet-last-fold/fistory_last.npy',allow_pickle=True).item()

time=np.load('/kaggle/input/densenet-last-fold/times_last.npy',allow_pickle=True).item()
print('Time taken for each fold for training all layers= '+str(np.mean(list(time.values()))))



print('Best mean results across all folds when training all layer is ='+str(np.mean(list(bst_acc.values()))))

print('Final mean results across all folds when training all layer is ='+str(np.mean(list(fnl_acc.values()))))


from matplotlib import pyplot as plt

for i in range(5):

    fold='fold_'+str(i+1)

    plt.plot(hist[fold]['loss'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Training all layers')

    plt.show()


from matplotlib import pyplot as plt

for i in range(5):

    fold='fold_'+str(i+1)

    plt.plot(hist[fold]['accuracy'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Training all layers')

    plt.show()


from matplotlib import pyplot as plt

for i in range(5):

    fold='fold_'+str(i+1)

    plt.plot(hist[fold]['val_loss'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.title('Training all layers')

    plt.show()


from matplotlib import pyplot as plt

for i in range(5):

    fold='fold_'+str(i+1)

    plt.plot(hist[fold]['val_accuracy'])

    plt.title('loss for fold '+str(i))

    plt.xlabel('epoch')

    plt.ylabel('accuracy')

    plt.title('Training all layer')

    plt.show()