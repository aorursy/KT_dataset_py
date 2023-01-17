!git clone https://github.com/miykael/gif_your_nifti
!pip install -r gif_your_nifti/requirements.txt
# Need to add the library to Python path

import sys

sys.path.insert(0, "gif_your_nifti")
!python gif_your_nifti/setup.py install
# Is it there?

!pip list | grep gif-your-nifti
# First, we need to move the .nii file to the output folder since it need to be writable 

# in addition to readable. 

!cp ../input/trends-assessment-prediction/fMRI_mask.nii fMRI_mask.nii
# Not the cleanest import but it works :p

from gif_your_nifti.core import write_gif_pseudocolor

size = 1

fps = 20

cmap = 'hot'

write_gif_pseudocolor("fMRI_mask.nii", size, fps, cmap)
# Is the .gif here?

!ls
from IPython.display import Image

Image("fMRI_mask_hot.gif", width=720, height=480)