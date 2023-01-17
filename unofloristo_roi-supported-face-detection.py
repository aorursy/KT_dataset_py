!pip install facenet_pytorch

!pip install opencv-contrib-python

from functools import wraps



from functools import wraps



import cv2

import numpy as np

from facenet_pytorch.models.mtcnn import MTCNN





def roi_supported_detector(dict_of_id_to_bbs, bbs_width):

    def mydecorator(detector_function):

        @wraps(detector_function)

        def wrapper(*args, **kwargs):

            person_id = kwargs['person_id']

            image = kwargs['img']

            if person_id in dict_of_id_to_bbs.keys():

                bb = dict_of_id_to_bbs[person_id][0]

                margin = 0.5

                width = abs(bb[0] - bb[2])

                roi_coords = np.asarray([max(bb[0] - width * margin, 0),

                                         max(bb[1] - width * margin, 0),

                                         min(bb[2] + width * margin, image.shape[1]),

                                         min(bb[3] + width * margin, image.shape[0])]).astype(int)

                roi = image[roi_coords[1]:roi_coords[3], roi_coords[0]:roi_coords[2], :]

                kwargs['img'] = roi

                cv2.imshow("img_roi", roi)

                result = detector_function(*args, **kwargs)

                result = list(result)

                if result[0] is not None:

                    result[0] = (result[0].reshape(-1, 2) + np.asarray(roi_coords)[:2]).reshape(1, -1)

                    if len(result) > 2:

                        result[2] = result[2] + np.asarray(roi_coords)[:2]

                    new_bb = result[0]

                    dict_of_id_to_bbs[person_id] = new_bb

                    bbs_width[person_id] = width



                result = tuple(result)

            else:

                result = detector_function(*args, **kwargs)

                if result[0] is not None:

                    new_bb = result[0]

                    dict_of_id_to_bbs[person_id] = new_bb

                    bb = dict_of_id_to_bbs[person_id][0]

                    width = abs(bb[0] - bb[2])

                    bbs_width[person_id] = width



            return result



        return wrapper



    return mydecorator





class Extended_MTCNN(MTCNN):

    dict_of_id_to_bbs = {}

    bbs_width = {}



    def __init__(self, *args, **kwargs):

        super().__init__(args, kwargs)



    @roi_supported_detector(dict_of_id_to_bbs, bbs_width)

    def detect_with_roi(self, person_id=1, *args, **kwargs):

        return self.detect(*args, **kwargs)

mtcnn = Extended_MTCNN(device=device,min_face_size=20)

cap = cv2.VideoCapture("path_to_video")

while ret:



    ret, frame = cap.read()

    person_id = 1

    

    if person_id in mtcnn.dict_of_id_to_bbs.keys():

        mtcnn.min_face_size = int(mtcnn.bbs_width[person_id] * 0.8)

        print(mtcnn.min_face_size)

    boxes, probs, landmarks = mtcnn.detect_with_roi(person_id=person_id,

                                                    img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), landmarks=True)