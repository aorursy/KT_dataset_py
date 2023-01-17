# Install facenet-pytorch

!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-2.0.0-py3-none-any.whl



from facenet_pytorch.models.inception_resnet_v1 import get_torch_home

torch_home = get_torch_home()



# Copy model checkpoints to torch cache so they are loaded automatically by the package

!mkdir -p $torch_home/checkpoints/

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth $torch_home/checkpoints/vggface2_DG3kwML46X.pt

!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth $torch_home/checkpoints/vggface2_G5aNV2VSMn.pt
import os

import glob

import time

import torch

import cv2

from PIL import Image

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm



# See github.com/timesler/facenet-pytorch:

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(f'Running on device: {device}')
# Load face detector

mtcnn = MTCNN(margin=14, keep_all=False,  select_largest=True, factor=0.5, device=device).eval()



# Load facial recognition model

resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
class DetectionPipeline:

    """Pipeline class for detecting faces in the frames of a video file."""

    

    def __init__(self, detector, batch_size=50, resize=None):

        """Constructor for DetectionPipeline class.

        

        Keyword Arguments:

            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})

            resize {float} -- Fraction by which to resize frames from original prior to face

                detection. A value less than 1 results in downsampling and a value greater than

                1 result in upsampling. (default: {None})

        """

        self.detector = detector

        self.batch_size = batch_size

        self.resize = resize

    

    def __call__(self, filename):

        """Load frames from an MP4 video and detect faces.



        Arguments:

            filename {str} -- Path to video.

        """

        # Create video reader and find length

        v_cap = cv2.VideoCapture(filename)

        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        

        sample = np.arange(0, v_len)

       

        # Loop through frames

        faces = []

        frames = []

        for j in range(50):

            success = v_cap.grab()

            if j in sample:

                # Load frame

                success, frame = v_cap.retrieve()

                if not success:

                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = Image.fromarray(frame)

                

                # Resize frame to desired size

                if self.resize is not None:

                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                frames.append(frame)



                # When batch is full, detect faces and reset frame list

                #if len(frames) % self.batch_size == 0 or j == sample[-1]:

                if j == 49:

                    faces.extend(self.detector(frames))

                    frames = []

                    v_cap.release()

                    return faces



def process_faces(faces, resnet):

    # Filter out frames without faces

    faces = [f for f in faces if f is not None]

    faces = torch.stack(faces).to(device)

    # use torch.cat to concatenate along the same dimension - if keepAll



    # Generate facial feature vectors using a pretrained model

    embeddings = resnet(faces)



    return embeddings.cpu().numpy()
# Define face detection pipeline

detection_pipeline = DetectionPipeline(detector=mtcnn, batch_size=60, resize=0.25)



# Get all test videos

filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')



X = []

start = time.time()

n_processed = 0

with torch.no_grad():

    for i, filename in tqdm(enumerate(filenames), total=len(filenames)):

        try:

            # Load frames and find faces

            faces = detection_pipeline(filename)

            embeddings = process_faces(faces, resnet)



            # Calculate embeddings

            X.append(embeddings)



        except KeyboardInterrupt:

            print('\nStopped.')

            break



        except Exception as e:

            print(e)

            X.append(None)

        

        n_processed += len(faces)

        print(f'Frames per second (load+detect+embed): {n_processed / (time.time() - start):6.3}\r', end='')
from tensorflow.keras.models import load_model



deepfake_detector = load_model('/kaggle/input/alldetector/model_all.hdf5')
def pad_sample(sample):

    result = np.zeros((50, 512))

    result[:sample.shape[0],:sample.shape[1]] = sample

    return result
submission = []

for filename, x_i in zip(filenames, X):

    if x_i is not None:

        x_i = pad_sample(x_i)

        x_i = np.expand_dims(x_i, axis=0)

        prob = deepfake_detector.predict(x_i)[0][0]

    else:

        prob = 0.5

    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])

submission.sort_values('filename').to_csv('submission.csv', index=False)



plt.hist(submission.label, 20)

plt.show()