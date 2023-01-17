from IPython.core.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))
import os

import sys 



facets_path = os.path.dirname('.')

facets_path = os.path.abspath(os.path.join(facets_path, 'facets', 'facets_overview', 'python'))

                              

if not facets_path in sys.path:

    sys.path.append(facets_path)
import PIL.Image

import pandas as pd

import numpy as np

import itertools

import math

import zipfile

from matplotlib import pylab as plt

%matplotlib inline
## Helpers for loading/viewing image data



# Background color for figures/subplots when showing favicons. Use a faint grey

# instead of the default white to make it clear which sections are transparent.

BG = '.95'

def show(df, scale=1, titles=None):

    """Show the favicons in the given dataframe, arranged in a grid.

    scale is a multiplier on the size of the drawn icons.

    """

    n = len(df)

    cols = int(min(n, max(4, 8//scale)))

    rows = math.ceil(n / cols)

    row_height = 1 * scale

    col_width = 1 * scale

    fs = (cols*col_width, rows*row_height)

    fig, axes = plt.subplots(rows, cols, figsize=fs, facecolor=BG)

    if rows == cols == 1:

        axes = np.array([axes])

    for i, (row, ax) in enumerate(

        itertools.zip_longest(df.itertuples(index=False), axes.flatten())

    ):

        if row is None:

            ax.axis('off')

        else:

            try:

                img = load_favicon(row.fname, row.split_index)

                _show_img(img, ax)

                if titles is not None:

                    ax.set_title(titles[i])

            except CorruptFaviconException:

                ax.axis('off')

                

def _show_img(img, ax=None):

    if ax is None:

        _fig, ax = plt.subplots(facecolor=BG)

    ax.tick_params(which='both', 

                   bottom=False, top=False, left=False, right=False,

                   labelbottom=False, labeltop=False, labelleft=False, labelright=False,

                  )

    ax.grid(False, which='both')

    plt.setp(list(ax.spines.values()), color='0.8', linewidth=1, linestyle='-')

    ax.set_facecolor(BG)

    cmap = None

    if img.mode in ('L', 'LA'):

        cmap = 'gray'

    ax.imshow(img, cmap=cmap, aspect='equal', interpolation='none')

                

class CorruptFaviconException(Exception): pass



_ZIP_LOOKUP = {}

def load_favicon(fname, split_ix):

    if split_ix not in _ZIP_LOOKUP:

        zip_fname = '../input/full-{}.z'.format(split_ix)

        _ZIP_LOOKUP[split_ix] = zipfile.ZipFile(zip_fname)

    archive = _ZIP_LOOKUP[split_ix]

    fp = archive.open(fname)

    try:

        fav = PIL.Image.open(fp)

    except (ValueError, OSError):

        raise CorruptFaviconException

    if fav.format == 'ICO' and len(fav.ico.entry) > 1:

        pil_ico_hack(fav)

    return fav



def pil_ico_hack(img):

    """When loading an ICO file containing multiple images, PIL chooses the

    largest. We want whichever one is listed first."""

    ico = img.ico

    ico.entry.sort(key = lambda d: d['offset'])

    first = ico.frame(0)

    first.load()

    img.im = first.im

    img.mode = first.mode

    img.size = first.size

def load_metadata_df():

    """Return a dataframe with a row of metadata for each favicon in the dataset."""

    csvpath = '../input/favicon_metadata.csv'

    return pd.read_csv(csvpath)
METADATA_CSV = load_metadata_df()

METADATA_CSV.head()
example_icons = METADATA_CSV.sample(6, random_state=123)

show(example_icons)



# That cat is adorable. Let's see a bigger version.

show(example_icons.iloc[5:6], scale=2)
import numpy as np

np.random.seed(2017)



indices = METADATA_CSV.index

indices = np.random.choice(indices, size=100000)

_metadata_csv = METADATA_CSV.loc[indices, :]
sys.path.append('/opt/facets/facets_overview/python')
from generic_feature_statistics_generator import GenericFeatureStatisticsGenerator

proto = GenericFeatureStatisticsGenerator().ProtoFromDataFrames([{'name': 'Metadata', 'table': _metadata_csv}])
from IPython.core.display import display, HTML

import base64

protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html" >

        <facets-overview id="elem"></facets-overview>

        <script>

          document.querySelector("#elem").protoInput = "{protostr}";

        </script>"""

html = HTML_TEMPLATE.format(protostr=protostr)

display(HTML(html))
sprit_image_size = (32, 32)

n = 300

ids = _metadata_csv[['fname', 'split_index']].values

m = int(np.ceil(len(ids) * 1.0 / n))

(m*sprit_image_size[0], n*sprit_image_size[1])
atlas_image_path = "complete_atlas_image.png"



sprit_image_size = (32, 32)

if not os.path.exists(atlas_image_path):    

    ids = _metadata_csv[['fname', 'split_index']].values

    n = 300

    m = int(np.ceil(len(ids) * 1.0 / n))

    complete_image = PIL.Image.new('RGBA', (n*sprit_image_size[0], m*sprit_image_size[1]))

    counter = 0

    for i in range(m):

        print("-- %i / %i" % (counter, len(ids)))

        ys = i*sprit_image_size[1]

        ye = ys + sprit_image_size[1]

        for j in range(n):

            xs = j*sprit_image_size[0]

            xe = xs + sprit_image_size[0]

            if counter == len(ids):

                break

            image_id = ids[counter]; counter+=1

            try:

                img = load_favicon(*image_id)

                if img.size != sprit_image_size:

                    img = img.resize(size=sprit_image_size, resample=PIL.Image.BICUBIC)

                complete_image.paste(img.convert(mode='RGBA'), (xs, ys))

            except Exception:

                pass            

        if counter == len(ids):

            break        

    

    complete_image.save(atlas_image_path)

    del complete_image
atlas_url = atlas_image_path
# Display the Dive visualization for this data

from IPython.core.display import display, HTML



HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html">

        <facets-dive 

            id="elem" 

            height="750"

            cross-origin="anonymous"

            sprite-image-width="32"

            sprite-image-height="32">

        </facets-dive>

        <script>

          var data = {jsonstr};

          var atlas_url = "{atlas_url}";

          document.querySelector("#elem").data = data;

          document.querySelector("#elem").atlasUrl = atlas_url;

        </script>"""

html = HTML_TEMPLATE.format(jsonstr=_metadata_csv.to_json(orient='records'), atlas_url=atlas_url)

display(HTML(html))
jsonstr=[{

  "name": "apple",

  "category": "fruit",

  "calories": 95

},{

  "name": "broccoli",

  "category": "vegetable",

  "calories": 50

}]
# Display the Dive visualization for this data

from IPython.core.display import display, HTML



HTML_TEMPLATE = """<link rel="import" href="/nbextensions/facets-jupyter.html">

        <facets-dive id="elem" height="750"></facets-dive>

        <script>

          var data = {jsonstr};

          document.querySelector("#elem").data = data;

        </script>"""

html = HTML_TEMPLATE.format(jsonstr=jsonstr)

display(HTML(html))