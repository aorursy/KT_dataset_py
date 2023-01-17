try:

    import numpy as np # linear algebra

    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt

    from collections import  Counter

    import seaborn as sns

    plt.style.use('ggplot')



    from sklearn.metrics import f1_score, confusion_matrix

    

    from nltk import word_tokenize

    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')

    import re

    import string

    import time

    

    print("Success: All Packages Loaded!")

except:

    print("Error: One or more packages failed to load.")
# Set your own project id here

try:

    PROJECT_ID = project_id = 'pivotal-shield-268206'

    from google.cloud import automl_v1beta1 as automl

    automl_client = client = automl.AutoMlClient()

    from google.cloud import storage

    storage_client = storage.Client(project=PROJECT_ID)

    print("Success: Connected to PROJECT_ID: {} \nUsing AutoML Client".format(project_id))

except:

    print("Error: Failed to Connect to AutoML Service!")
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

data = df = train = pd.read_csv("../input/nlp-getting-started/train.csv")



print("Success:\n[1]Test file loaded as `test`\n[2]Train file loaded as `df`,`data`,`train`\n[3]Sample Submission file loaded as `sample_submmission`")
train.head(10)
test.head(10)
print("Rows = {}, Colums = {}".format(data.shape[0], data.shape[1]))

shape1 = (data.shape[0], data.shape[1])

data = data[data['text'].notnull()]

if data.shape == shape1: 

    print ('Data Consistent') 

else: 

    print ('Data Inconsistent')
sns.set_style('whitegrid')

x = data.target.value_counts()

sns.barplot(x.index,x)

plt.gca().set_ylabel('samples')

print('0: Not Disaster Tweets, 1: Disaster Tweets')
"""

The functions to clean the text are defined below. 

Removes URL, HTML, emojis, possessive case, punctuations, and converts to lower case. 

"""



def removeURL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def removeHtml(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def removePossessive(text):

    word = re.compile(r"\'s")

    return word.sub(r'',text)

    

def removeEmoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def removePunct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def allLower(text):

    word = text.lower()

    return word
df['text']=df['text'].apply(removeURL)

print("[CLEAN] Removing URLs")

time.sleep(0.5)

df['text']=df['text'].apply(removeHtml)

print("[CLEAN] Removing HTML Tags")

time.sleep(0.5)

df['text']=df['text'].apply(removeEmoji)

print("[CLEAN] Removing Emoticons")

time.sleep(0.5)

df['text']=df['text'].apply(removePossessive)

print("[CLEAN] Removing Possessive Pronouns")

time.sleep(0.5)

df['text']=df['text'].apply(removePunct)

print("[CLEAN] Removing Punctuations")

time.sleep(0.5)

df['text']=df['text'].apply(allLower)

print("[CLEAN] Converting all to Lower Case")

time.sleep(0.5)

print("Success: Text Cleaned!")
df.head()
df.drop(['id','keyword','location'], axis=1, inplace=True)
#Uploads a Data Frame (df) to given bucketName on GCS with given fileName

def uploadToBucket(df, fileName, bucketName):

    df.to_csv(fileName, index=False)

    bucket = storage_client.get_bucket(bucketName)

    blob = bucket.blob(fileName)

    blob.upload_from_filename(fileName)
#Set the display_name of Dataset for GCP UI.

display_name = "realornot_kaggle_train_clean"

project_location = client.location_path(PROJECT_ID, "us-central1")



# MultiClass: At most one label is allowed per example.

metadata = automl.types.TextClassificationDatasetMetadata(

    classification_type=automl.enums.ClassificationType.MULTICLASS

)

dataset = automl.types.Dataset(

    display_name=display_name,

    text_classification_dataset_metadata=metadata,

)
# Create a dataset with the dataset metadata in the region.

def createDatasetAutoML(project_location, dataset):

    response = client.create_dataset(project_location, dataset)
#Add Dataset ID Manually from the AutoML UI

dataset_id = "TCN1801116594529632256"

path = "gs://realornot_kaggle_nirvanaai/train_clean.csv"

dataset_full_id = client.dataset_path(

    project_id, "us-central1", dataset_id

)
#Imports data from the Bucket on GCS (stored in csv format) to Dataset on GCP. 

def addData(path,dataset_full_id):

    try:

        input_uris = path.split(",")

        gcs_source = automl.types.GcsSource(input_uris=input_uris)

        input_config = automl.types.InputConfig(gcs_source=gcs_source)

        response = client.import_data(dataset_full_id, input_config)

        print("SUCCESS: Request Sent\n\nProcessing text items...\nThis can take several minutes or more. You will be emailed once importing has completed.")

    except:

        print("ERROR: Something went wrong. Cannot process Request.")
#Initializing Model Parameters

metadata = automl.types.TextClassificationModelMetadata()

model = automl.types.Model(

    display_name=display_name,

    dataset_id=dataset_id,

    text_classification_model_metadata=metadata,

)

model_full_id = client.model_path(project_id, "us-central1", model_id)
#Sends the Request to train the model on given dataset. You can check the progress on GCP AutoML UI.

def trainAutoMLModel(project_location, model):    

    #Create a model with the model metadata in the region.

    response = client.create_model(project_location, model)

    print(u"Training operation name: {}".format(response.operation.name))

    print("Training started...")
#Gives a list of all the trained model for a particular project_id

def listModels(project_id):

    project_location = client.location_path(project_id, "us-central1")

    response = client.list_models(project_location, "")



    print("List of models:")

    for model in response:

        # Display the model information.

        if (

            model.deployment_state

            == automl.enums.Model.DeploymentState.DEPLOYED

        ):

            deployment_state = "deployed"

        else:

            deployment_state = "undeployed"



        print("Model name: {}".format(model.name))

        print("Model id: {}".format(model.name.split("/")[-1]))

        print("Model display name: {}".format(model.display_name))

        print("Model create time:")

        print("\tseconds: {}".format(model.create_time.seconds))

        print("\tnanos: {}".format(model.create_time.nanos))

        print("Model deployment state: {}".format(deployment_state))
#Gives detailed information about the selected model (through model_id) of the given project.

def describeModel(model_id, project_id):

    model_full_id = client.model_path(project_id, "us-central1", model_id)



    print("List of model evaluations:")

    for evaluation in client.list_model_evaluations(model_full_id, ""):

        print("Model evaluation name: {}".format(evaluation.name))

        print(

            "Model annotation spec id: {}".format(

                evaluation.annotation_spec_id

            )

        )

        print("Create Time:")

        print("\tseconds: {}".format(evaluation.create_time.seconds))

        print("\tnanos: {}".format(evaluation.create_time.nanos / 1e9))

        print(

            "Evaluation example count: {}".format(

                evaluation.evaluated_example_count

            )

        )

        print(

            "Classification model evaluation metrics: {}".format(

                evaluation.classification_evaluation_metrics

            )

        )
#Returns the prediction score for the given content (text string) using a selected model (model_full_id) 

prediction_client = automl.PredictionServiceClient()

def predictAutoML(content, model_full_id=model_full_id):

    score = {'0': 0, '1': 0}

    text_snippet = automl.types.TextSnippet(

        content=content, mime_type="text/plain"

    )

    payload = automl.types.ExamplePayload(text_snippet=text_snippet)



    response = prediction_client.predict(model_full_id, payload)



    for annotation_payload in response.payload:

        className = annotation_payload.display_name

        if className == '0':

            score['0'] = annotation_payload.classification.score

        elif className == '1':

            score['1'] = annotation_payload.classification.score

    

    return score
#Takes a data frame as an argument and Returns a dictionary of id and label associated with it.

def getPredictionScores(data):

    test['target'] = np.nan

    sub = {}

    res = []

    for content in test.iterrows():

        score = predictAutoML(str(content[1]['text']))

        time.sleep(1)

        if float(score['0'])>float(score['1']):

            label = '0'

        else:

            label = '1'

        #time.sleep(1)

        sub = {'id': content[1]['id'], 'target': label}

        res.append(sub)

        print ("Success! id: {} labeled as: {}".format(content[1]['id'], label))

    print("Success!")

    return(res)
"""

The main code block. Uncomment # and run one by one.

getPredictionScores(data) will return a dictionary with id and labels.

"""

##Creating the dataset



#createDatasetAutoML(project_location, dataset)

#uploadToBucket(df, 'train_clean.csv', 'realornot_kaggle_nirvanaai')

#addData(path, dataset_full_id)

#trainAutoMLModel(project_location, model)



#listModels(project_id)

model_id = "TCN161333540166828032"

#describeModel(model_id, project_id)

##After Model is Trained get the model_id from the above request.

predictions = getPredictionScores(test)



#TESTING

model_id = "TCN161333540166828032"

model_full_id = client.model_path(project_id, "us-central1", model_id)

content = "We're shaking...It's an earthquake"

print(predictAutoML(content, model_full_id))
final = pd.DataFrame.from_dict(predictions)
final[['id', 'target']].to_csv('[VN]submissionAutoML.csv', index=False)