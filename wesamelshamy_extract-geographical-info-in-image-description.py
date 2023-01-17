from PIL import Image
from io import BytesIO
import piexif
import warnings
warnings.filterwarnings('ignore')

exif_dict = piexif.load('../input/exif-data/68f7904482b7ef4c.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)
exif_dict = piexif.load('../input/exif-data/81cb67de72768b6f.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)
exif_dict = piexif.load('../input/exif-data/6072c8a91b1a6f54.jpg')
image_desc = exif_dict['0th'][piexif.ImageIFD.ImageDescription]
print(image_desc)
# -*- coding: utf-8 -*-

# !/usr/bin/python

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import tqdm
import piexif


def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def download_image(key_url):
    out_dir = sys.argv[2]
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
        # Read Exif data if the image has it.
        # Otherwise, create empty exif_bytes
        if pil_image.info.get('exif', None):
            exif_dict = piexif.load(pil_image.info['exif'])
            exif_bytes = piexif.dump(exif_dict)
        else:
            exif_bytes = piexif.dump({})
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        # Include exif data with saved file.
        pil_image_rgb.save(filename, format='JPEG', quality=90, exif=exif_bytes)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=20)  # Num of CPUs
    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list), total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    if os.environ['PYTHONPATH'] == '/kaggle/lib/kagglegym':
        print('This script does not run as intented on Kaggle.  Downlaod and run locally.')
    else:
        loader()