import pydicom

import glob

from pathlib import Path



def read_dicom_dir(dicom_path):

    """Reads a folder containing multiple .dcm files from a single patient"""

    dcm_files = dicom_path.glob('*.dcm')

    dicom_files = [pydicom.dcmread(str(x)) for x in dcm_files] 

    return dicom_files
tmp = read_dicom_dir(Path('../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184'))

print(tmp[0])
tmp[0].SliceThickness # Verifying the dataset is v2, which added the SliceThickness tag
! md5sum ../input/osic-pulmonary-fibrosis-progression/train/ID00011637202177653955184/1.dcm