!pip install --upgrade pip
!pip install -q tensorflow-io
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re

#color
from colorama import Fore, Back, Style

import tensorflow as tf
import tensorflow_io as tfio

#read dicom
import pydicom

#visialisation
from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import matplotlib

#Pandas Profiling
import pandas_profiling as pdp

#For segmentation and 3d plotting
import skimage
import skimage.measure as measure
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc

# Settings for pretty nice plots
plt.style.use('seaborn-bright')
plt.show()

dataDir = '../input/osic-pulmonary-fibrosis-progression'
workDir = '../input/output/'
train_data = pd.read_csv(os.path.join(dataDir, 'train.csv'))
train_data.head()
train_data.info()
# Function to plot characteristics of patients
def plot_columns_data(df, col, bins):
    plt.figure(figsize = (8, 4))
    plt.hist(df[col], bins = bins)
    plt.xlabel('{}'.format(col))
    plt.ylabel('Count')
    try:
        plt.text(50, 450, r'$\mu={},\ \sigma={}$'.format(df[col].unique().mean().round(2), df[col].unique().std().round(2)), fontsize = 12)
    except:
        pass
    plt.show() 
print(Fore.YELLOW + 'People\'s ages, sorted: ', Style.RESET_ALL, sorted(train_data.Age.unique()))
print(Fore.YELLOW + 'Youngest person\'s age:', Style.RESET_ALL, train_data.Age.unique().min())
print(Fore.YELLOW + 'Oldest person\'s age  :', Style.RESET_ALL, train_data.Age.unique().max())
print(Fore.YELLOW + 'Mean age of people in the dataset:', Style.RESET_ALL, train_data.Age.unique().mean().round(2))
print(Fore.YELLOW + 'Person\'s sex  : ', Style.RESET_ALL, train_data.Sex.unique())
print(Fore.YELLOW + 'Smoking status: ', Style.RESET_ALL, train_data.SmokingStatus.unique())
plot_columns_data(train_data, 'Age', 7)
plot_columns_data(train_data, 'Sex', 2)
train_data.groupby(['Sex']).count()['Patient'].to_frame()
plot_columns_data(train_data, 'SmokingStatus', 3)
train_data.groupby(['SmokingStatus']).count()['Patient'].to_frame()
print('There are', Fore.YELLOW + '{}'.format(len(train_data["Patient"].unique())), Style.RESET_ALL, 'unique patients in Train Data.', "\n")

# Recordings per Patient
data = train_data.groupby("Patient")["Weeks"].count().reset_index(drop=False)
# Sort by Weeks
data = data.sort_values(['Weeks']).reset_index(drop=True)
print('Minimum recordings per patient:', Fore.YELLOW + '{}'.format(data["Weeks"].min()), Style.RESET_ALL, "\n" +
      'Maximum recordings per patient:', Fore.YELLOW + '{}'.format(data["Weeks"].max()), Style.RESET_ALL)
print('Min FVC value:', Fore.YELLOW + '{}'.format(train_data.FVC.unique().min()), Style.RESET_ALL)
print('Max FVC value:', Style.RESET_ALL, Fore.YELLOW + '{}'.format(train_data.FVC.unique().max()), Style.RESET_ALL)
print('Min Percent value:', Fore.MAGENTA + '{} %'.format(train_data.Percent.unique().min().round(2)), Style.RESET_ALL)
print('Max Percent value:', Fore.MAGENTA + '{} %'.format(train_data.Percent.unique().max().round(2)), Style.RESET_ALL)
# Figure
fig, (fvc, pct) = plt.subplots(1, 2, figsize = (16, 6))

fvc = sns.distplot(train_data["FVC"], ax=fvc, color = 'orange')
pct = sns.distplot(train_data["Percent"], ax=pct, color = 'blue')

fvc.set_title("FVC Distribution", fontsize=16)
pct.set_title("Percent Distribution", fontsize=16);
print('Min no. weeks pre(negative number)/post base CT scan:', Fore.YELLOW + '{}'.format(train_data['Weeks'].min()), Style.RESET_ALL)
print('Max no. weeks pre(negative number)/post base CT scan:', Fore.YELLOW + '{}'.format(train_data['Weeks'].max()), Style.RESET_ALL)
plt.figure(figsize = (16, 6))

w = sns.distplot(train_data['Weeks'])
plt.title("Number of weeks before/after the CT scan", fontsize = 16)
plt.xlabel("Weeks", fontsize=14);
#Let's check the correlations
plt.figure(figsize=(16,6))
sns.heatmap(train_data.corr(), 
            annot=True, 
            linewidths = .5,
            square=True,
            annot_kws={'size':12, 'weight': 'bold'},
            center = 0,
            cmap=sns.color_palette('bright'))
plt.yticks(rotation = 0)
plt.show()
test_data = pd.read_csv(os.path.join(dataDir, 'test.csv'))
# test_data.head()
sample_sub = pd.read_csv(os.path.join(dataDir, 'sample_submission.csv'))
#sample_sub.head()
trainDicomPath = '../input/osic-pulmonary-fibrosis-progression/train'
testDicomPath = '../input/osic-pulmonary-fibrosis-progression/test'
def getListOfFiles(dirName):
    ''' For the given path, get the List of 
        all files in the directory tree 
    '''
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles   
# Get the list of all train files in directory tree at given path
listOfTrainDcmFiles = getListOfFiles(trainDicomPath)
len(listOfTrainDcmFiles)
# Get the list of all test files in directory tree at given path
listOfTestDcmFiles = getListOfFiles(testDicomPath)
len(listOfTestDcmFiles)
#Show random scans
random.seed(22)
for file in random.sample(listOfTrainDcmFiles, 3):
    dataset = pydicom.dcmread(file)
    
    print(Fore.MAGENTA + "Patient id.......:", Style.RESET_ALL, dataset.PatientID, "\n" +
          Fore.MAGENTA + "Modality.........:", Style.RESET_ALL, dataset.Modality, "\n" +
          Fore.MAGENTA + "Rows.............:", Style.RESET_ALL, dataset.Rows, "\n" +
          Fore.MAGENTA + "Columns..........:", Style.RESET_ALL, dataset.Columns)
    
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()
    
patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"
datasets = []

# First Order the files in the dataset
files = []
for dcm in list(os.listdir(patient_dir)):
    files.append(dcm) 
files.sort(key=lambda f: int(re.sub('\D', '', f)))

# Read in the Dataset
for dcm in files:
    path = patient_dir + "/" + dcm
    datasets.append(pydicom.dcmread(path))

# Plot the images
fig=plt.figure(figsize=(16, 6))
columns = 10
rows = 3

for i in range(1, columns*rows +1):
    img = datasets[i-1].pixel_array
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.title(i, fontsize = 9)
    plt.axis('off');
# Create base director for Train .dcm files
director = "../input/osic-pulmonary-fibrosis-progression/train"

# Create path column with the path to each patient's CT
train_data["Path"] = director + "/" + train_data["Patient"]

# Create variable that shows how many CT scans each patient has
train_data["CT_number"] = 0

for k, path in enumerate(train_data["Path"]):
    train_data["CT_number"][k] = len(os.listdir(path))
train_data.head()
def create_gif(number_of_CT = 87):
    """Picks a patient at random and creates a GIF with their CT scans."""
    
    # Select one of the patients
    # patient = "ID00007637202177411956430"
    patient = train_data[train_data["CT_number"] == number_of_CT].sample(random_state=1)["Patient"].values[0]
    
    # === READ IN .dcm FILES ===
    patient_dir = "../input/osic-pulmonary-fibrosis-progression/train/" + patient
    datasets = []

    # First Order the files in the dataset
    files = []
    for dcm in list(os.listdir(patient_dir)):
        files.append(dcm) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Read in the Dataset from the Patient path
    for dcm in files:
        path = patient_dir + "/" + dcm
        datasets.append(pydicom.dcmread(path))
        
        
    # === SAVE AS .png ===
    # Create directory to save the png files
    if os.path.isdir(f"png_{patient}") == False:
        os.mkdir(f"png_{patient}")

    # Save images to PNG
    for i in range(len(datasets)):
        img = datasets[i].pixel_array
        matplotlib.image.imsave(f'png_{patient}/img_{i}.png', img)
        
        
    # === CREATE GIF ===
    # First Order the files in the dataset (again)
    files = []
    for png in list(os.listdir(f"../working/png_{patient}")):
        files.append(png) 
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Create the frames
    frames = []

    # Create frames
    for file in files:
    #     print("../working/png_images/" + name)
        new_frame = Image.open(f"../working/png_{patient}/" + file)
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(f'gif_{patient}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=200, loop=0)
create_gif(number_of_CT=12)
create_gif(number_of_CT=30)
create_gif(number_of_CT=87)

print("First file len:", len(os.listdir("../working/png_ID00165637202237320314458")), "\n" +
      "Second file len:", len(os.listdir("../working/png_ID00199637202248141386743")), "\n" +
      "Third file len:", len(os.listdir("../working/png_ID00340637202287399835821")))
show_gif(filename="./gif_ID00165637202237320314458.gif", format='png', width=400, height=400)
show_gif(filename="./gif_ID00340637202287399835821.gif", format='png', width=400, height=400)
show_gif(filename="./gif_ID00199637202248141386743.gif", format='png', width=400, height=400)
profile_train_df = pdp.ProfileReport(train_data)
profile_train_df
def read_ct_scan(folder_name):
        # Read the slices from the dicom file
        slices = [pydicom.dcmread(folder_name + filename) for filename in os.listdir(folder_name)]
        
        # Sort the dicom slices in their respective order
        slices.sort(key=lambda x: int(x.InstanceNumber))
        
        # Get the pixel values for all the slices
        slices = np.stack([s.pixel_array for s in slices])
        slices[slices == -2000] = 0
        return slices
ct_scan_1 = read_ct_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/') 
ct_scan_2 = read_ct_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00047637202184938901501/')
ct_scan_3 = read_ct_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/')
ct_scan_4 = read_ct_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00062637202188654068490/')

# ct_scan_2.shape
def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(10, 10))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
plot_ct_scan(ct_scan_2)
def get_segmented_lungs(im, plot=False):
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < 604
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    if plot == True:
        plots[7].axis('off')
        plots[7].imshow(im, cmap=plt.cm.gray) 
        
    return im

def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])
def further_remove_noise(segmented_ct_scan):
    selem = ball(2)
    binary = binary_closing(segmented_ct_scan, selem)

    label_scan = label(binary)

    areas = [r.area for r in regionprops(label_scan)]
    areas.sort()

    for r in regionprops(label_scan):
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = 1000, 1000, 1000
    
        for c in r.coords:
            max_z = max(c[0], max_z)
            max_y = max(c[1], max_y)
            max_x = max(c[2], max_x)
        
            min_z = min(c[0], min_z)
            min_y = min(c[1], min_y)
            min_x = min(c[2], min_x)
        if (min_z == max_z or min_y == max_y or min_x == max_x or r.area > areas[-3]):
            for c in r.coords:
                segmented_ct_scan[c[0], c[1], c[2]] = 0
        else:
            index = (max((max_x - min_x), (max_y - min_y), (max_z - min_z))) / (min((max_x - min_x), (max_y - min_y) , (max_z - min_z)))
segmented_ct_scan_1 = segment_lung_from_ct_scan(ct_scan_1)
segmented_ct_scan_1[segmented_ct_scan_1 < 604] = 0
further_remove_noise(segmented_ct_scan_1)
# plot_ct_scan(segmented_ct_scan_1)
segmented_ct_scan_2 = segment_lung_from_ct_scan(ct_scan_2)
segmented_ct_scan_2[segmented_ct_scan_2 < 604] = 0
further_remove_noise(segmented_ct_scan_2)
# plot_ct_scan(segmented_ct_scan_2)
segmented_ct_scan_3 = segment_lung_from_ct_scan(ct_scan_3)
segmented_ct_scan_3[segmented_ct_scan_3 < 604] = 0
further_remove_noise(segmented_ct_scan_3)
#plot_ct_scan(segmented_ct_scan_3)
segmented_ct_scan_4 = segment_lung_from_ct_scan(ct_scan_4)
segmented_ct_scan_4[segmented_ct_scan_4 < 604] = 0
further_remove_noise(segmented_ct_scan_4)
#plot_ct_scan(segmented_ct_scan_4)
def plot_3d(image):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces, normals, values = measure.marching_cubes_lewiner(p)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    #     ax.set_xlim(0, p.shape[0])
    #     ax.set_ylim(0, p.shape[1])
    #     ax.set_zlim(0, p.shape[2])
    
    ax.set_xlim(np.min(verts[:,0]), np.max(verts[:,0]))
    ax.set_ylim(np.min(verts[:,1]), np.max(verts[:,1])) 
    ax.set_zlim(np.min(verts[:,2]), np.max(verts[:,2]))
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    
    face_color = [0.45, 0.45, 0.75]    
    mesh.set_facecolor(face_color)
    
    mesh.set_edgecolor('k')
    
    ax.add_collection3d(mesh)
    
    plt.tight_layout()
    plt.show()
plot_3d(segmented_ct_scan_1)
train_data.loc[train_data.Patient == 'ID00007637202177411956430'].mean()
plot_3d(segmented_ct_scan_2)
train_data.loc[train_data.Patient == 'ID00047637202184938901501'].mean()
plot_3d(segmented_ct_scan_3)
train_data.loc[train_data.Patient == 'ID00012637202177665765362'].mean()
plot_3d(segmented_ct_scan_4)
train_data.loc[train_data.Patient == 'ID00062637202188654068490'].mean()





