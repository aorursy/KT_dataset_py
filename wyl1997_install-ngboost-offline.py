import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

!mkdir -p /tmp/pip/cache

!cp /kaggle/input/ngboost_whl_depend/autograd-1.3.xyz /tmp/pip/cache/autograd-1.3.tar.gz

!pip install --no-index --find-links /tmp/pip/cache/ autograd

!ls /kaggle/input/ngboost_whl_depend/

%cd /kaggle/input/ngboost_whl_depend/

!pip install autograd_gamma-0.4.1-py3-none-any.whl lifelines-0.23.6-py3-none-any.whl numpy-1.18.1-cp36-cp36m-manylinux1_x86_64.whl scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl ngboost-0.1.4-py3-none-any.whl

%cd /kaggle/working
# !mkdir -p /tmp/pip/cache

# !cp /kaggle/input/ngboost-whl-new/ngboost_whl_depend/autograd-1.3.xyz /tmp/pip/cache/autograd-1.3.tar.gz

# !pip install --no-index --find-links /tmp/pip/cache/ autograd

# !ls /kaggle/input/ngboost-whl-new/ngboost_whl_depend/

# %cd /kaggle/input/ngboost-whl-new/ngboost_whl_depend/

# !pip install autograd_gamma-0.4.1-py3-none-any.whl lifelines-0.23.6-py3-none-any.whl numpy-1.18.1-cp36-cp36m-manylinux1_x86_64.whl scipy-1.4.1-cp36-cp36m-manylinux1_x86_64.whl ngboost-0.1.4-py3-none-any.whl

# %cd /kaggle/working
import ngboost as ngb
