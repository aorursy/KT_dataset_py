import os

import csv

import pandas as pd

import pydicom as dicom
%%time

train_folder_dcm = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'

train_images_path = os.listdir(train_folder_dcm)



test_folder_dcm = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'

test_images_path = os.listdir(test_folder_dcm)
%%time

dataset = dicom.dcmread(train_folder_dcm + train_images_path[0])

dataset
%%time

dataset = dicom.dcmread(test_folder_dcm + test_images_path[0])

dataset
# list of attributes in dicom image

attributes = ['SOPInstanceUID', 'Modality', 'PatientID', 'StudyInstanceUID',

              'SeriesInstanceUID', 'StudyID', 'ImagePositionPatient',

              'ImageOrientationPatient', 'SamplesPerPixel', 'PhotometricInterpretation',

              'Rows', 'Columns', 'PixelSpacing', 'BitsAllocated', 'BitsStored', 'HighBit',

              'PixelRepresentation', 'WindowCenter', 'WindowWidth', 'RescaleIntercept',

              'RescaleSlope', 'PixelData']
%%time

with open('train_patient_detail.csv', 'w', newline ='') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')

    writer.writerow(attributes)

    for image in train_images_path:

        ds = dicom.dcmread(os.path.join(train_folder_dcm, image))

        rows = []

        for field in attributes:

            if ds.data_element(field) is None:

                rows.append('')

            else:

                x = str(ds.data_element(field)).replace("'", "")

                y = x.find(":")

                x = x[y+2:]

                rows.append(x)

        writer.writerow(rows)
%%time

with open('test_patient_detail.csv', 'w', newline ='') as csvfile:

    writer = csv.writer(csvfile, delimiter=',')

    writer.writerow(attributes)

    for image in test_images_path:

        ds = dicom.dcmread(os.path.join(test_folder_dcm, image))

        rows = []

        for field in attributes:

            if ds.data_element(field) is None:

                rows.append('')

            else:

                x = str(ds.data_element(field)).replace("'", "")

                y = x.find(":")

                x = x[y+2:]

                rows.append(x)

        writer.writerow(rows)