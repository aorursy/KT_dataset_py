"""

from google.cloud import storage

from zipfile import ZipFile

from tqdm import tqdm



# Get the bucket

def getstoragebucket():

    # Create a client instance

    storage_client = storage.Client()

    # Get bucket name

    bucket = storage_client.get_bucket('<bucket name>')

    return bucket



# read zip files and copy on storage bucket

def copy_images_on_data_storage(filename):

    bucket = getstoragebucket()

    with ZipFile(filename, 'r') as myzip:

        for contentfilename in tqdm(myzip.namelist()):

            contentfile = myzip.read(contentfilename)

            # create a blob with same hierarchy

            blob = bucket.blob(contentfilename)

            blob.upload_from_string(contentfile)

            

if __name__ == '__main__':

    filename = f'<absolute path of data .zip file'

    # Example /home/<username>/.../<.zip file name>

    copy_images_on_data_storage(filename)

    print('Successfully copied all file in storage bucket')

"""