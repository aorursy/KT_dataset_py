import pydicom

filename = "/kaggle/input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/ID_6431af929.dcm"

with pydicom.dcmread(filename) as ds:

    im = ds.pixel_array