!pip install natsort
import os

from natsort import natsorted
def get_filenames(path):

    st_fname = []

    for f_name in os.listdir(f'{path}'):

        st_fname.append(f_name)

    return st_fname
path = '../input/chai-time-data-science/Cleaned Subtitles/'
episode_files = get_filenames(path)
type(episode_files)
episode_files
episode_id = natsorted(episode_files)
episode_id
list1 = ['episode69.1.Masala', 'episode63.2.Paan_rose', 'episode48.3.Ginger2', 'episode44.4.Herbal', 'episode35.5.Ginger1']

list2 = ['sanyam', 2.75, 'E49', '4.12', 'E27', 29.33, 6.89, 'jeremy']
natsorted(list1)
natsorted(list2)