!wget https://www.tbi.univie.ac.at/RNA/download/ubuntu/ubuntu_18_04/viennarna_2.4.15-1_amd64.deb
!apt-get install ./viennarna_2.4.15-1_amd64.deb -y
!git clone https://github.com/DasLab/arnie

!/opt/conda/bin/python3.7 -m pip install --upgrade pip
!git clone https://www.github.com/DasLab/draw_rna draw_rna_pkg
!cd draw_rna_pkg && python setup.py install
import os
import sys

!echo "vienna_2: /usr/bin" > arnie.conf
!echo "TMP: /kaggle/working/tmp" >> arnie.conf
!mkdir -p /kaggle/working/tmp
os.environ["ARNIEFILE"] = f"/kaggle/working/arnie.conf"
sys.path.append('/kaggle/working/draw_rna_pkg/')
sys.path.append('/kaggle/working/draw_rna_pkg/ipynb/')
%load_ext autoreload
%autoreload 2
%pylab inline

import seaborn as sns
sns.set_context('poster')
sns.set_style('white')
import numpy as np
from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from decimal import Decimal
sequence = "CGCUGUCUGUACUUGUAUCAGUACACUGACGAGUCCCUAAAGGACGAAACAGCG"
mfe_structure = mfe(sequence)
print(mfe_structure)
bp_matrix = bpps(sequence)
plt.imshow(bp_matrix, origin='lower left', cmap='gist_heat_r')

from ipynb.draw import draw_struct
draw_struct(sequence, mfe_structure)
p_unp_vec = 1 - np.sum(bp_matrix, axis=0)
plot(p_unp_vec)
xlabel('Sequence position')
ylabel('p(unpaired)')
draw_struct(sequence, mfe_structure, c = p_unp_vec, cmap='plasma')