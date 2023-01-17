import os

import pathlib



!pip install -q tensorflow==1.14.0



# Turn on internet access!

!git clone https://github.com/lassebuurrasmussen/bacteriocin_classifier
os.chdir('bacteriocin_classifier')
# Thanks to https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import requests



def download_file_from_google_drive(id, destination):

    URL = "https://docs.google.com/uc?export=download"



    session = requests.Session()



    response = session.get(URL, params = { 'id' : id }, stream = True)

    token = get_confirm_token(response)



    if token:

        params = { 'id' : id, 'confirm' : token }

        response = session.get(URL, params = params, stream = True)



    save_response_content(response, destination)    



def get_confirm_token(response):

    for key, value in response.cookies.items():

        if key.startswith('download_warning'):

            return value



    return None



def save_response_content(response, destination):

    CHUNK_SIZE = 32768



    with open(destination, "wb") as f:

        for chunk in response.iter_content(CHUNK_SIZE):

            if chunk: # filter out keep-alive new chunks

                f.write(chunk)



file_id = '1VaA92XizlP88AjJTPr7BJ2Nh_q2qmi5s'

destination = pathlib.Path('data/elmo_model_uniref50/seqvec.zip')

destination.parent.mkdir(exist_ok=True)

download_file_from_google_drive(file_id, destination)



file_id = '1UaFuCirtm289Y6Q7dbKCxpPI1AEtix8g'

destination = pathlib.Path('code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006/cnnpar_weights.zip')

destination.parent.mkdir(exist_ok=True)

download_file_from_google_drive(file_id, destination)

print(os.listdir("data/elmo_model_uniref50"), os.listdir("code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006/"))
!unzip 'code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006/cnnpar_weights.zip' -d 'code_modules/nn_training/BAC_UNI_len2006/final_elmo_CNNPAR_BAC_UNI_len2006/'

!unzip "data/elmo_model_uniref50/seqvec.zip" -d "data/elmo_model_uniref50/"

os.rename("data/elmo_model_uniref50/uniref50_v2/options.json", "data/elmo_model_uniref50/options.json")

os.rename("data/elmo_model_uniref50/uniref50_v2/weights.hdf5", "data/elmo_model_uniref50/weights.hdf5")
%run run_model.py sample_fasta.faa results.csv
!cat 'results.csv'


os.rename("results.csv", "../results.csv")

os.chdir("../")

!rm -rf bacteriocin_classifier