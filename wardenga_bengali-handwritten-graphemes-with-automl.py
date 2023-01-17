# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_labels=pd.read_csv('../input/bengaliai-cv19/train.csv')

test_labels=pd.read_csv('../input/bengaliai-cv19/test.csv')

class_map=pd.read_csv('../input/bengaliai-cv19/class_map.csv')

sample_submission=pd.read_csv('../input/bengaliai-cv19/sample_submission.csv')
class_map.head()
train_labels.head()
# Set your own project id here

PROJECT_ID = 'noble-return-265322'

BUCKET_REGION = 'us-central1'#europe-west3 is frankfurt but other 

                             #regions than the first are not supp'd  



from google.cloud import storage



storage_client = storage.Client(project=PROJECT_ID)



# The name for the new bucket

BUCKET_NAME = 'bengaliai'



# Creates the new bucket

bucket=storage.Bucket(storage_client,name=BUCKET_NAME)

if not bucket.exists():

    bucket.create(location=BUCKET_REGION)



print("Bucket {} created.".format(BUCKET_NAME))
#to upload data (datapoint=blob) to a bucket



def upload_blob(bucket_name, source_file_name, destination_blob_name,printyes=False):

    """Uploads a file to the bucket."""

    # bucket_name = "your-bucket-name"

    # source_file_name = "local/path/to/file"

    # destination_blob_name = "storage-object-name"

    

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(destination_blob_name)



    blob.upload_from_filename(source_file_name)

    

    if printyes:

        print(

            "File {} uploaded to {}.".format(

                source_file_name, destination_blob_name

                )

            )

        

def download_to_kaggle(bucket_name,destination_directory,file_name,prefix=None):

    """

    Takes the data from your GCS Bucket and puts it

    into the working directory of your Kaggle notebook

    """

    os.makedirs(destination_directory, exist_ok = True)

    full_file_path = os.path.join(destination_directory, file_name)

    blobs = storage_client.list_blobs(bucket_name,prefix=prefix)

    for blob in blobs:

        blob.download_to_filename(full_file_path)
HEIGHT = 137

WIDTH = 236

N_CHANNELS=1
i=0 

name=f'train_image_data_{i}.parquet'

train_img = pd.read_parquet('../input/bengaliai-cv19/'+name)
train_img.shape
train_img.head()
# Visualize few samples of current training dataset

from matplotlib import pyplot as plt



fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

count=0

for row in ax:

    for col in row:

        col.imshow(train_img.iloc[[count]].drop(['image_id'],axis=1).to_numpy(dtype=np.float32).reshape(HEIGHT, WIDTH).astype(np.float64),cmap='binary')

        count += 1

plt.show()
from matplotlib.image import imsave

img=train_img.iloc[1]

image_path=img.image_id+'.png'

imsave(image_path,img.drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64))



upload_blob(BUCKET_NAME,image_path,image_path,printyes=True)
from matplotlib.image import imsave

from os import remove



def upload_bengaliai_data(path):

    train_img = pd.read_parquet(path)

    num=0

    shape=train_img.shape

    for i in range(shape[0]):

        img=train_img.iloc[i]

        image_path=img.image_id+'.png'    

        #create a .png out of the columns and save under theimage_id

        imsave(image_path,img.drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64))

        

        #upload to the bucket

        if num%1000==0:

            upload_blob(BUCKET_NAME,image_path,image_path,printyes=True)

        else:

            upload_blob(BUCKET_NAME,image_path,image_path)

        

        #delete the file in the working directory

        remove(image_path)

        

        num+=1            
download_data=False

if download_data:

    for i in range(4):

        print(f'staring with file{i}...')

        upload_bengaliai_data(f'/kaggle/input/bengaliai-cv19/train_image_data_{i}.parquet')

        print(f'file {i} finished.')

    

    for i in range(4):

        print(f'staring with file{i}...')

        upload_bengaliai_data(f'/kaggle/input/bengaliai-cv19/test_image_data_{i}.parquet')

        print(f'file {i} finished.')
import csv



BUCKET_LINK='gs://'+BUCKET_NAME



BUCKET_NAME='bengaliai'

BUCKET_LINK='gs://'+BUCKET_NAME

train_labels['uri']=[BUCKET_LINK+'/'+image_id+'.png' for image_id in train_labels['image_id']]

train_labels['g']=['g'+str(num) for num in train_labels['grapheme_root']]

train_labels['v']=['v'+str(num) for num in train_labels['vowel_diacritic']]

train_labels['c']=['c'+str(num) for num in train_labels['consonant_diacritic']]

labels=train_labels.drop(['image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme'],axis=1)

labels.to_csv('all_data.csv',header=False,index=False)
all_data=pd.read_csv('all_data.csv')

all_data.head()
all_data.shape
#set up the AutoMl client



from google.cloud import automl_v1beta1 as automl

#automl_client = automl.AutoMlClient() #not working at the moment



from google.api_core.gapic_v1.client_info import ClientInfo

automl_client = automl.AutoMlClient(client_info=ClientInfo())



display_name='bengaliai_dataset'



# A resource that represents Google Cloud Platform location.

project_location = automl_client.location_path(PROJECT_ID, BUCKET_REGION)
dataset_names=[]

for dataset in automl_client.list_datasets(project_location):

    dataset_names.append(dataset.name)

    print(dataset.name)
new_dataset=False

try:

    response = automl_client.get_dataset(name=dataset_names[0])

    print('loading successfull.')

except:

    print('couldn\'t get Dataset. Creating new Dataset')

    new_dataset = True

    #Specify the classification type

    #Types:

    #MultiLabel: Multiple labels are allowed for one example.

    #MultiClass: At most one label is allowed per example.

    metadata = automl.types.ImageClassificationDatasetMetadata(classification_type=automl.enums.ClassificationType.MULTILABEL)

    dataset = automl.types.Dataset(display_name=display_name,image_classification_dataset_metadata=metadata)

    response = automl_client.create_dataset(project_location, dataset)

    

    # Create a dataset with the dataset metadata in the region.

print("Dataset name: {}".format(response.name))

print("Dataset id: {}".format(response.name.split("/")[-1]))

print("Dataset display name: {}".format(response.display_name))

print("Image classification dataset metadata:")

print("\t{}".format(dataset.image_classification_dataset_metadata))

print("Dataset example count: {}".format(response.example_count))

print("Dataset create time:")

print("\tseconds: {}".format(response.create_time.seconds))

print("\tnanos: {}".format(response.create_time.nanos))
DATASET_ID=response.name.split("/")[-1]
# Get the full path of the dataset.=response.name

dataset_full_id = automl_client.dataset_path(PROJECT_ID, 'us-central1', DATASET_ID)
all_data_path = 'gs://' + BUCKET_NAME + '/all_data.csv'



input_uris = all_data_path.split(",")

input_config = {"gcs_source": {"input_uris": input_uris}}





import_data=False #set to true if you havent imported

if import_data:

    response=automl_client.import_data(name=response.name,input_config=input_config)



    print("Processing import...")

    # synchronous check of operation status.

    print("Data imported. {}".format(response.result()))
# Set model name and model metadata for the image dataset.

TRAIN_BUDGET = 1 # (specified in hours, from 1-100 (int))

MODEL_NAME='bengaliai'
models=automl_client.list_models(project_location)

model_names=[]

for md in models:

    model_names.append(md.name)

    print(md.name)
model=None



try:

    model=automl_client.get_model(model_names[0])

    print('loaded the model {}'.format(model.name))

except:

    model_params= {

    "display_name": MODEL_NAME,

    "dataset_id": DATASET_ID,

    "image_classification_model_metadata": {"train_budget": TRAIN_BUDGET}

    if TRAIN_BUDGET

    else {},}

    print('loading model unscucessfull.')

    print('creating new model')

    response=automl_client.create_model(project_location,model_params)

    print("Training operation name: {}".format(response.operation.name))

    print("Training started...")

    

    #wait till training is done

    model=response.result()

print('Model name: {}'.format(model.name))

print(print("Model id: {}".format(model.name.split("/")[-1])))



#save the model_id for further use as with the dataset

MODEL_FULL_ID=model.name

MODEL_ID=model.name.split("/")[-1]
print('List of model evaluations:')

num=0#

for evaluation in automl_client.list_model_evaluations(MODEL_FULL_ID, ''):

    if num<=1:

        #take this evaluation and show some metric within        

        response = automl_client.get_model_evaluation(evaluation.name)



        print(u'Model evaluation name: {}'.format(response.name))

        print(u'Model annotation spec id: {}'.format(response.annotation_spec_id))

        print('Create Time:')

        print(u'\tseconds: {}'.format(response.create_time.seconds))

        print(u'\tnanos: {}'.format(response.create_time.nanos / 1e9))

        print(u'Evaluation example count: {}'.format(

            response.evaluated_example_count))

        print('Classification model evaluation metrics: {}'.format(

            response.classification_evaluation_metrics))

        num=num+1

    

    
automl_client.list_model_evaluations(MODEL_FULL_ID, '')

#response=automl_client.deploy_model(model.name) #uncomment if not deployed before
#we need to set up a prediction client 

prediction_client = automl.PredictionServiceClient(client_info=ClientInfo())



#read the file to be predicted

import io

def bengali_make_predict(images):

    """

    Returns a prediction of the grapheme components of the images in 'images'

    -images=pd.dataframe containing the image as rows with first column being the image_id

    """

    for i in range(images.shape[0]):

        #convert the rows to the correct size np.array

        img=images.iloc[i].drop(['image_id']).to_numpy(dtype=np.float64).reshape(HEIGHT, WIDTH).astype(np.float64)

        

        #create a stream as to not save the image in the workspace and pass directly

        imageBytearray=io.BytesIO()

        imsave(imageBytearray,img,format='png')

        

        image=automl.types.Image(image_bytes=imageBytearray.getvalue())

        payload=automl.types.ExamplePayload(image=image)

        

        #define some parameters of the model

        params={'score_threshold': '0.8'}

        

        response=prediction_client.predict(MODEL_FULL_ID,payload, params)

        print('Prediction results:')

        for result in response.payload:

            print(u'Predicted class name: {}'.format(result.display_name))

            print(u'Predicted class score: {}'.format(result.classification.score))
test_images=pd.read_parquet('../input/bengaliai-cv19/test_image_data_0.parquet')
test_images.shape
bengali_make_predict(test_images)