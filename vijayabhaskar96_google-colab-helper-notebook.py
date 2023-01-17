!pip install git+https://github.com/Kaggle/kaggle-api.git --upgrade
import os
credentials = {"username":"vijayabhaskar96","key":"123456a45847983a4537dbae3f23d612f"}
os.environ['KAGGLE_USERNAME']=credentials["username"]
os.environ['KAGGLE_KEY']=credentials["key"]
!kaggle competitions download -c jovian-pytorch-z2g
!unzip jovian-pytorch-z2g.zip
DATA_DIR = '/content/Human protein atlas'

TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images
TEST_DIR = DATA_DIR + '/test'                             # Contains test images

TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images
TEST_CSV = '/content/submission.csv'   # Contains dummy labels for test image