!pip install ../input/facenet-pytorch-vggface2/facenet_pytorch-2.0.1-py3-none-any.whl



import sys, os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import glob

import time

import torch

import random

import time

from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face

from torchvision.transforms import Normalize, RandomHorizontalFlip, ToTensor, ToPILImage, Compose, Resize

from sklearn.metrics import log_loss

import pathlib

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
class Video_reader:

    def extract_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []

        while(cap.isOpened()):

            ret, frame = cap.read()

            if ret==True:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frames.append(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):

                    break

            else:

                break

        cap.release()

        assert len(frames) != 0

        return np.array(frames)

        

        

    def extract_one_frame(self, video_path, frame_index):

        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  #设置要获取的帧号

        _, frame=cap.read()

        cap.release()

        if _:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            return frame

        else:

            return None



class Face_extractor:

    def __init__(self):

        pass

        

    def _get_boundingbox(self, bbox, width, height, scale=1.2, minsize=None):

        x1, y1, x2, y2 = bbox[:4]

        if not 0.33 < (x2-x1)/(y2-y1) < 3:

            return np.array([0,0,0,0])

        size_bb = int(max(x2 - x1, y2 - y1) * scale)

        if minsize:

            if size_bb < minsize:

                size_bb = minsize

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2



        x1 = max(int(center_x - size_bb / 2), 0)

        y1 = max(int(center_y - size_bb / 2), 0)

        size_bb = min(width - x1, size_bb)

        size_bb = min(height - y1, size_bb)



        return np.array([x1,y1,x1+size_bb,y1+size_bb]).astype(int)

    

    

    def _rectang_crop(self, image, bbox):

        height, width = image.shape[:2]

        l,t,r,b = self._get_boundingbox(bbox, width, height) 

        return image[t:b, l:r]

    

    def _get(images):

        pass

    

    

    def get_faces(self, images, with_person_num = False, only_one = True):

        faces, nums = self._get(images)

        if only_one:

            faces = [face[0] for face in faces if len(face)>0]

            nums = [num for num, face in zip(nums,faces) if len(face)>0]

        if with_person_num:

            faces = (faces, nums)

        return faces

    

    

    def get_face(self, image, with_person_num = False, only_one = True):

        faces, nums = self.get_faces(np.array([image]), with_person_num=True, only_one=False)

        faces, nums = faces[0], nums[0]

        if only_one:

            if len(faces)>0:

                faces = faces[0]

            else:

                faces = None

        if with_person_num:

            faces = (faces, nums)

        return faces

    

    

class MTCNN_extractor(Face_extractor):

    def __init__(self, device = 'cuda:0' if torch.cuda.is_available() else 'cpu', down_sample = 2):

        self.extractor = MTCNN(keep_all=True, device=device, min_face_size=80//down_sample).eval()

        self.down_sample = down_sample

            

    def _get(self, images):

        h, w = images.shape[1:3]

        pils = [Image.fromarray(img).resize((w//self.down_sample, h//self.down_sample)) for img in images]

        bboxes, probs = self.extractor.detect(pils)

        

        facelist = [[self._rectang_crop(img, box) for box in boxes*self.down_sample] for boxes, img in zip(bboxes,images) if boxes is not None]

        person_nums = [np.sum(prob>0.9) for prob,fss in zip(probs, bboxes) if fss is not None]

        

        assert len(person_nums) == len(facelist)

        return facelist, person_nums

    

    

class Inference_model:

    def __init__(self):

        pass

        

    

    def data_transform(self):

        # transform to Tensor

        pre_trained_mean, pre_trained_std = [0.439, 0.328, 0.304], [0.232, 0.206, 0.201]

        return Compose([Resize(224), ToTensor(), Normalize(pre_trained_mean, pre_trained_std)])

    

    

    def TTA(self, pil_img):

        return [pil_img, RandomHorizontalFlip(p=1)(pil_img)]

    

    

    def predict(self, batch):

        return 0.5

    

    def test(self, shape = (1,3,224,224)):

        return self.predict(torch.rand(shape))
def predict_on_all(video_paths, video_lables = [], video_reader=None, 

                   face_extractor=None, models=None, sample_number = 13):

    if video_reader is None:

        video_reader = Video_reader()

    if face_extractor is None:

        face_extractor = MTCNN_extractor()

    if models is None:

        models = [Inference_model()]



    def predict_one_video(file_path):

        with torch.no_grad():

            try:

                import time

                start = time.time()

                frames = video_reader.extract_video(file_path)

                sample = np.linspace(0, len(frames) - 1, sample_number*2).round().astype(int)

                faces = face_extractor.get_faces(frames[sample])

                np.random.shuffle(faces)

                if len(faces) == 0:

                    print("no face detected")

                    return 0.5

                pils = [Image.fromarray(face)  for face in faces[:sample_number] ]



                answers = []

                for model in models:

                    tr = model.data_transform()

                    batch = torch.stack([tr(p) for img in pils for p in model.TTA(img)])

                    answers.append(model.predict(batch))

                return np.mean(answers)

            except Exception as e:

                print("Error with ", file_path, e)

                return 0.5



    predicts = [predict_one_video(i) for i in tqdm(video_paths)]

    

    if len(video_lables) == len(video_paths):

        print(f"loss = {log_loss(video_lables, predicts, labels=[0,1])}")

    return predicts
kaggle = True

speed_test = True

print(f"using {'cuda:0' if torch.cuda.is_available() else 'cpu'}")



filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

labels = []



models = [Inference_model()]

face_extractor = MTCNN_extractor()
# from pympler.tracker import SummaryTracker

# tracker = SummaryTracker()



if speed_test:

    testnum = 5

    start = time.time()

    test_filenames, test_labels = filenames[:testnum], []

    ret = predict_on_all(test_filenames, video_lables=test_labels, models=models, face_extractor=face_extractor)

    time_dur = time.time()-start

    print(f"totally {time_dur} s used, {time_dur/testnum} s per video")
predictions = predict_on_all(filenames, video_lables=labels, models=models, face_extractor=face_extractor)

print(np.mean(predictions))



submission_df = pd.DataFrame({"filename": [fn.split('/')[-1] for fn in filenames], "label": predictions})

submission_df.to_csv("submission.csv", index=False)