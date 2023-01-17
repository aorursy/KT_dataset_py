# Add-ons

# Google Cloud Services



# Data: Add data

# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# https://www.kaggle.com/alexisbcook/automl-tables-wrapper



# Settings

# Internet On
# Is this cell needed?



!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex_automl import *

step_1.check()
"""

o All we need is a project id --- that has a billing information ---



There is Google Cloud Platform Free Tier.

I have previously signed up for the free trial -- its 12-month period has passed, and no remaining of the $300 in credit left.

But Always Free allows the use of AutoML Tables up to 6 node hours.



Upgrading allows to access Always Free, which activates the billing information.

Then, create a new project tied with that billing info.

GCP set a project id that should be specified in the cell below.



o Activate APIs as well (Step 4 in https://www.kaggle.com/alexisbcook/get-started-with-google-cloud-platform).

"""
# This cell is a Code Snippet from Add-ons: Google Cloud Services



# Set your own project id here

#PROJECT_ID = 'your-google-cloud-project'

PROJECT_ID = 'always-free-kaggle'



#The followings are covered by AutoMLTablesWrapper.prepare_clients() called in the next cell



#from google.cloud import automl_v1beta1 as automl

#automl_client = automl.AutoMlClient()

#from google.cloud import storage

#storage_client = storage.Client(project=PROJECT_ID)
# AutoML Tables' Free trial is 6 hours. Use 2 hours of them for the House Prices competition (remaining 4 hours for the Tutorial). 



# TODO: Fill in your project ID and bucket name

#PROJECT_ID = 'your-project-id-here'

#BUCKET_NAME = 'your-bucket-name-here'

BUCKET_NAME = 'learn-intro-to-ml-8-intro-to-automl-exercise' # No uppercase letters



# Do not change: Fill in the remaining variables

DATASET_DISPLAY_NAME = 'house_prices_dataset' # underscore

TRAIN_FILEPATH = "../input/house-prices-advanced-regression-techniques/train.csv"

TEST_FILEPATH = "../input/house-prices-advanced-regression-techniques/test.csv"

TARGET_COLUMN = 'SalePrice'

ID_COLUMN = 'Id'

MODEL_DISPLAY_NAME = 'house_prices_model' # underscore

TRAIN_BUDGET = 2000



# Do not change: Create an instance of the wrapper

from automl_tables_wrapper import AutoMLTablesWrapper



amw = AutoMLTablesWrapper(project_id=PROJECT_ID,

                          bucket_name=BUCKET_NAME,

                          dataset_display_name=DATASET_DISPLAY_NAME,

                          train_filepath=TRAIN_FILEPATH,

                          test_filepath=TEST_FILEPATH,

                          target_column=TARGET_COLUMN,

                          id_column=ID_COLUMN,

                          model_display_name=MODEL_DISPLAY_NAME,

                          train_budget=TRAIN_BUDGET)
# Do not change: Create and train the model

amw.train_model()
# Do not change: Get predictions

amw.get_predictions()