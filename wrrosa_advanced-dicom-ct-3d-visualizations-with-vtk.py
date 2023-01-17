! pip install pyvirtualdisplay -q

! apt-get install -y xvfb >> /dev/null



from IPython.display import Image

import imageio

import os

import shutil

import matplotlib.pyplot as plt 

import matplotlib.image as mpimg 

import gc

from vtk.util import numpy_support

import numpy

from pyvirtualdisplay import Display



disp = Display().start()

import vtk

disp.stop()



N =  18

default_width = 512

default_height = 512



def vtk_show(renderer, width = default_width, height = default_height, filename = ""):



    renderWindow = vtk.vtkRenderWindow()

    

    renderWindow.SetOffScreenRendering(1)

    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(width, height)

    renderWindow.Render()

     

    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(renderWindow)

    windowToImageFilter.Update()

     

    writer = vtk. vtkPNGWriter()

    

    if filename == "":

        writer.SetWriteToMemory(1)

        writer.SetInputConnection(windowToImageFilter.GetOutputPort())

        writer.Write()    

        return bytes(memoryview(writer.GetResult()))

    else:

        writer.SetFileName(filename+".png")

        writer.SetInputConnection(windowToImageFilter.GetOutputPort())

        writer.Write()    

        return None

    

def vtk_render_gif(renderer, N, name, Roll = False, Azimuth = False, Elevation = False, Actor = None, RotateX = False, RotateY = False, RotateZ = False, Zoom = 0, Dolly = 0, standard = True, width = default_width, height = default_height):    

    if standard:

        renderer.ResetCamera()

        camera = renderer.MakeCamera()

        renderer.ResetCameraClippingRange()

        camera.SetPosition(0,0,0)

    os.makedirs(name,exist_ok=True)

    

    if Zoom != 0:

        renderer.GetActiveCamera().Zoom(Zoom)

        

    if Dolly != 0:

        renderer.GetActiveCamera().Dolly(Dolly)

        

    #tmpN = 1

    if N >0: # render gif

        for fi in range(N):

            if Roll:

                renderer.GetActiveCamera().Roll(360//N) 

            if Azimuth:

                renderer.GetActiveCamera().Azimuth(360//N) 

            if Elevation:

                renderer.GetActiveCamera().Elevation(360//N)

            if Actor is not None:

                if RotateX:

                    Actor.RotateX(360//N)

                if RotateY:

                    Actor.RotateY(360//N)

                if RotateZ:

                    Actor.RotateZ(360//N)                    

            vtk_show(renderer,filename = name + "/shot"+str(fi), width = width, height = height)

        # render gif and cleanup

        img_list = []

        for fi in range(N):

            img_list.append(mpimg.imread(name + '/shot' + str(fi) + '.png'))

        shutil.rmtree(name)

        imageio.mimsave(name + ".gif", img_list, duration=0.5)



    #if N == 1: # render png

       #vtk_show(renderer,filename = name + ".gif")



def CreateLut():

    colors = vtk.vtkNamedColors()



    colorLut = vtk.vtkLookupTable()

    colorLut.SetNumberOfColors(17)

    colorLut.SetTableRange(0, 16)

    colorLut.Build()



    colorLut.SetTableValue(0, 0, 0, 0, 0)

    colorLut.SetTableValue(1, colors.GetColor4d("salmon"))  # blood

    colorLut.SetTableValue(2, colors.GetColor4d("beige"))  # brain

    colorLut.SetTableValue(3, colors.GetColor4d("orange"))  # duodenum

    colorLut.SetTableValue(4, colors.GetColor4d("misty_rose"))  # eye_retina

    colorLut.SetTableValue(5, colors.GetColor4d("white"))  # eye_white

    colorLut.SetTableValue(6, colors.GetColor4d("tomato"))  # heart

    colorLut.SetTableValue(7, colors.GetColor4d("raspberry"))  # ileum

    colorLut.SetTableValue(8, colors.GetColor4d("banana"))  # kidney

    colorLut.SetTableValue(9, colors.GetColor4d("peru"))  # l_intestine

    colorLut.SetTableValue(10, colors.GetColor4d("pink"))  # liver

    colorLut.SetTableValue(11, colors.GetColor4d("powder_blue"))  # lung

    colorLut.SetTableValue(12, colors.GetColor4d("carrot"))  # nerve

    colorLut.SetTableValue(13, colors.GetColor4d("wheat"))  # skeleton

    colorLut.SetTableValue(14, colors.GetColor4d("violet"))  # spleen

    colorLut.SetTableValue(15, colors.GetColor4d("plum"))  # stomach



    return colorLut



def CreateTissueMap():

    tissueMap = dict()

    tissueMap["blood"] = 1

    tissueMap["brain"] = 2

    tissueMap["duodenum"] = 3

    tissueMap["eyeRetina"] = 4

    tissueMap["eyeWhite"] = 5

    tissueMap["heart"] = 6

    tissueMap["ileum"] = 7

    tissueMap["kidney"] = 8

    tissueMap["intestine"] = 9

    tissueMap["liver"] = 10

    tissueMap["lung"] = 11

    tissueMap["nerve"] = 12

    tissueMap["skeleton"] = 13

    tissueMap["spleen"] = 14

    tissueMap["stomach"] = 15



    return tissueMap



tissueMap = CreateTissueMap()



colorLut = CreateLut()



def CreateTissue(reader, ThrIn, ThrOut, color = "skeleton", isoValue = 127.5):

    selectTissue = vtk.vtkImageThreshold()

    selectTissue.ThresholdBetween(ThrIn,ThrOut)

    selectTissue.ReplaceInOn()

    selectTissue.SetInValue(255)

    selectTissue.ReplaceOutOn()

    selectTissue.SetOutValue(0)

    selectTissue.Update()

    selectTissue.SetInputConnection(reader.GetOutputPort())



    gaussianRadius = 5

    gaussianStandardDeviation = 2.0

    gaussian = vtk.vtkImageGaussianSmooth()

    gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)

    gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)

    gaussian.SetInputConnection(selectTissue.GetOutputPort())



    #isoValue = 127.5

    mcubes = vtk.vtkMarchingCubes()

    mcubes.SetInputConnection(gaussian.GetOutputPort())

    mcubes.ComputeScalarsOff()

    mcubes.ComputeGradientsOff()

    mcubes.ComputeNormalsOff()

    mcubes.SetValue(0, isoValue)



    smoothingIterations = 5

    passBand = 0.001

    featureAngle = 60.0

    smoother = vtk.vtkWindowedSincPolyDataFilter()

    smoother.SetInputConnection(mcubes.GetOutputPort())

    smoother.SetNumberOfIterations(smoothingIterations)

    smoother.BoundarySmoothingOff()

    smoother.FeatureEdgeSmoothingOff()

    smoother.SetFeatureAngle(featureAngle)

    smoother.SetPassBand(passBand)

    smoother.NonManifoldSmoothingOn()

    smoother.NormalizeCoordinatesOn()

    smoother.Update()



    normals = vtk.vtkPolyDataNormals()

    normals.SetInputConnection(smoother.GetOutputPort())

    normals.SetFeatureAngle(featureAngle)



    stripper = vtk.vtkStripper()

    stripper.SetInputConnection(normals.GetOutputPort())



    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(stripper.GetOutputPort())



    actor = vtk.vtkActor()

    actor.SetMapper(mapper)

    actor.GetProperty().SetColor( colorLut.GetTableValue(tissueMap[color])[:3])

    actor.GetProperty().SetSpecular(.5)

    actor.GetProperty().SetSpecularPower(10)

    

    return actor



def render_lungs(workdir, datadir, patient):

    PathDicom = datadir + patient

    reader = vtk.vtkDICOMImageReader()

    reader.SetDirectoryName(PathDicom)

    reader.Update()    

    disp = Display().start()

    renderer = vtk.vtkRenderer()

    actor = CreateTissue(reader,-2000,-300,"lung", isoValue = 170)

    renderer.AddActor(actor)

    renderer.SetBackground(1.0, 1.0, 1.0)



    renderer.ResetCamera()

    renderer.ResetCameraClippingRange()

    camera = renderer.GetActiveCamera()

    camera.Elevation(120)

    camera.Elevation(120)

    renderer.SetActiveCamera(camera)



    name = workdir + patient + '_lungs'



    vtk_render_gif(renderer, 1, name, Dolly = 1.5,width = 400, height = 400)

    disp.stop()

    gc.collect()
## supporting lines =  tidying up

workdir = '/kaggle/working/patients/'

os.makedirs(workdir, exist_ok = True)

datadir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"

patients = os.listdir(datadir)

patients.sort()

patient = patients[17]



## vtk reading dicom

reader = vtk.vtkDICOMImageReader()

reader.SetDirectoryName(datadir + patient)

reader.Update()
windowing = {}

windowing['lungs'] = [1500,-600,64,123,147]

windowing['mediastinum'] = [350,50,255,244,209]

windowing['bones'] = [300,400,177,122,101]

windowing['blood'] = [5,80,216,101,79]



patient = "ID00012637202177665765362"

reader.SetDirectoryName(datadir+patient)

reader.Update()



imageData = reader.GetOutput()

volumeMapper = vtk.vtkSmartVolumeMapper()

volumeMapper.SetInputData(imageData)

volumeProperty = vtk.vtkVolumeProperty()

volumeProperty.SetInterpolationType(vtk.VTK_LINEAR_INTERPOLATION)



for cur_windowing in windowing:

    cur_w = windowing[cur_windowing]

    opacity_function = vtk.vtkPiecewiseFunction()

    opacity_function.AddPoint(cur_w[1]-cur_w[0]/2,   0.0)

    opacity_function.AddPoint(cur_w[1],   1.0)

    opacity_function.AddPoint(cur_w[1]+cur_w[0]/2,   0.0)

    volumeProperty.SetScalarOpacity(opacity_function)



    color_function = vtk.vtkColorTransferFunction()

    color_function.SetColorSpaceToDiverging()

    color_function.AddRGBPoint(cur_w[1]-cur_w[0]/2,0,0,0)

    color_function.AddRGBPoint(cur_w[1],cur_w[2],cur_w[3],cur_w[4])

    color_function.AddRGBPoint(cur_w[1]+cur_w[0]/2, 0,0,0)



    volumeProperty.SetColor(color_function)



    volume = vtk.vtkVolume()

    volume.SetMapper(volumeMapper)

    volume.SetProperty(volumeProperty)



    disp = Display().start()

    renderer = vtk.vtkRenderer();

    volumeMapper.SetRequestedRenderModeToRayCast()

    renderer.AddViewProp(volume)

    

    renderer.ResetCamera()

    renderer.SetBackground(1,1,1);

    renderer.ResetCamera()

    renderer.ResetCameraClippingRange()

    camera = renderer.MakeCamera()



    camera.SetPosition(0,0,0)

    camera = renderer.GetActiveCamera()

    camera.Dolly(1.5)





    camera.Roll(360)

    name = workdir + patient + cur_windowing + '_top'

    vtk_render_gif(renderer, N = 1 ,name =  name, standard = False)

    

    name = workdir + patient + cur_windowing + '_front'

    camera.Elevation(240)

    camera.Elevation(20)

    vtk_render_gif(renderer, N = 1 ,name =  name, standard = False)

    

    disp.stop()

    

plt.rcParams["figure.figsize"] = (40,40)

idp = 0 

for cur_windowing in windowing:

    idp += 1

    plt.subplot(len(windowing),4, idp)

    try:

        im = mpimg.imread( workdir + patient + cur_windowing+'_top.gif') 

        plt.imshow(im) 

        plt.title('Windowing: ' + cur_windowing, fontsize =20)

    except:

        pass
plt.rcParams["figure.figsize"] = (40,40)

idp = 0 

for cur_windowing in windowing:

    idp += 1

    plt.subplot(len(windowing),4, idp)

    try:

        im = mpimg.imread( workdir + patient + cur_windowing+'_front.gif') 

        plt.imshow(im) 

        plt.title('Windowing: ' + cur_windowing, fontsize =20)

    except:

        pass
%%time



patient = patients[17]

reader = vtk.vtkDICOMImageReader()

reader.SetDirectoryName(datadir+patient)

reader.Update()



disp = Display().start()

renderer = vtk.vtkRenderer()

renderer.AddActor(CreateTissue(reader,-900,-400,"lung"))

renderer.AddActor(CreateTissue(reader,0,120,"blood"))

renderer.AddActor(CreateTissue(reader,100,2000,"skeleton"))

renderer.SetBackground(1.0, 1.0, 1.0)



renderer.ResetCamera()

renderer.ResetCameraClippingRange()

camera = renderer.GetActiveCamera()

camera.Elevation(120)

camera.Roll(180)

renderer.SetActiveCamera(camera)



name = workdir + patient + "_front"

vtk_render_gif(renderer, 1, name, Dolly = 1.5)

disp.stop()



Image(filename=name + ".gif", format='png')    
%%time

disp = Display().start()

renderer = vtk.vtkRenderer()

renderer.AddActor(CreateTissue(reader,-900,-400,"lung"))

renderer.AddActor(CreateTissue(reader,0,120,"blood"))

renderer.AddActor(CreateTissue(reader,100,2000,"skeleton"))



renderer.SetBackground(1.0, 1.0, 1.0)



renderer.ResetCamera()

renderer.ResetCameraClippingRange()

camera = renderer.GetActiveCamera()

camera.Elevation(120)

camera.Elevation(120)

camera.Roll(180)

renderer.SetActiveCamera(camera)



name = workdir + patient + "_back"

vtk_render_gif(renderer, 1, name, Dolly = 1.5)

disp.stop()



Image(filename=name + ".gif", format='png')    
%%time

disp = Display().start()

renderer = vtk.vtkRenderer()

actor = CreateTissue(reader,-2000,-300,"lung", isoValue = 170)

renderer.AddActor(actor)

renderer.SetBackground(1.0, 1.0, 1.0)



renderer.ResetCamera()

renderer.ResetCameraClippingRange()

camera = renderer.GetActiveCamera()

camera.Elevation(120)

camera.Elevation(120)

renderer.SetActiveCamera(camera)



name = workdir + patient + '_lungs'



vtk_render_gif(renderer, 1, name, Dolly = 1.5)

disp.stop()



Image(filename=name + ".gif", format='png')
%%time

s_patients = [patients[i] for i in [1,2,4,6,8,12,13,14,15,16]]

for patient in s_patients:

    try:

        render_lungs(workdir, datadir, patient)

        print(patient + ' completed render lungs')

    except:

        print(patient + ' failed render lungs')



plt.rcParams["figure.figsize"] = (40,120)

idp = 0 

for patient in s_patients:

    idp += 1

    plt.subplot(10,2, idp)

    try:

        im = mpimg.imread( workdir + patient + '_lungs.gif') 

        plt.imshow(im) 

        plt.title('OSIC PatientID: '+patient, fontsize=20)

    except:

        pass
from vtk.util import numpy_support

import numpy



reader = vtk.vtkDICOMImageReader()

reader.SetDirectoryName(datadir + patient)

reader.Update()

# Load dimensions using `GetDataExtent`

_extent = reader.GetDataExtent()

ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]



# Load spacing values

ConstPixelSpacing = reader.GetPixelSpacing()



# Get the 'vtkImageData' object from the reader

imageData = reader.GetOutput()

# Get the 'vtkPointData' object from the 'vtkImageData' object

pointData = imageData.GetPointData()

# Ensure that only one array exists within the 'vtkPointData' object

assert (pointData.GetNumberOfArrays()==1)

# Get the `vtkArray` (or whatever derived type) which is needed for the `numpy_support.vtk_to_numpy` function

arrayData = pointData.GetArray(0)



# Convert the `vtkArray` to a NumPy array

ArrayDicom = numpy_support.vtk_to_numpy(arrayData)

# Reshape the NumPy array to 3D using 'ConstPixelDims' as a 'shape'

ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')

ArrayDicom.shape
## source https://www.kaggle.com/allunia/pulmonary-dicom-preprocessing



basepath = ""

from os import listdir

import pydicom

import numpy as np



def load_scans(dcm_path):

    if basepath == "../input/osic-pulmonary-fibrosis-progression/":

        # in this competition we have missing values in ImagePosition, this is why we are sorting by filename number

        files = listdir(dcm_path)

        file_nums = [np.int(file.split(".")[0]) for file in files]

        sorted_file_nums = np.sort(file_nums)[::-1]

        slices = [pydicom.dcmread(dcm_path + "/" + str(file_num) + ".dcm" ) for file_num in sorted_file_nums]

    else:

        # otherwise we sort by ImagePositionPatient (z-coordinate) or by SliceLocation

        slices = [pydicom.dcmread(dcm_path + "/" + file) for file in listdir(dcm_path)]

        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    return slices

def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)



    #images = set_outside_scanner_to_air(images)

    

    # convert to HU

    for n in range(len(slices)):

        

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)

scans = load_scans(datadir + patient)

hu_scans = transform_to_hu(scans)

np.mean(hu_scans) == np.mean(ArrayDicom)
### if needed:

#shiftScale = vtk.vtkImageShiftScale()

#shiftScale.SetScale(reader.GetRescaleSlope())

#shiftScale.SetShift(reader.GetRescaleOffset())

#shiftScale.SetInputConnection(reader.GetOutputPort())

#shiftScale.Update()
! pip install pyvirtualdisplay -q

from pyvirtualdisplay import Display



disp = Display().start()

import vtk

disp.stop()
vtk.vtkVersion().GetVTKMajorVersion()
def vtk_show(renderer, width = default_width, height = default_height):



    renderWindow = vtk.vtkRenderWindow()

    

    renderWindow.SetOffScreenRendering(1)

    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(width, height)

    renderWindow.Render()

     

    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(renderWindow)

    windowToImageFilter.Update()

     

    writer = vtk.vtkPNGWriter()

    writer.SetWriteToMemory(1)

    writer.SetInputConnection(windowToImageFilter.GetOutputPort())

    writer.Write()    

    

    return bytes(memoryview(writer.GetResult()))
disp = Display().start()
colors = vtk.vtkNamedColors()

# Set the background color.

bkg = map(lambda x: x / 255.0, [26, 51, 102, 255])

colors.SetColor("BkgColor", *bkg)
cylinder = vtk.vtkCylinderSource()

cylinder.SetResolution(8)
cylinderMapper = vtk.vtkPolyDataMapper()

cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
cylinderActor = vtk.vtkActor()

cylinderActor.SetMapper(cylinderMapper)

cylinderActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))

cylinderActor.RotateX(30.0)

cylinderActor.RotateY(-45.0)
ren = vtk.vtkRenderer()

# useless lines: in kaggle notebook

#renWin = vtk.vtkRenderWindow() 

#renWin.AddRenderer(ren)

#iren = vtk.vtkRenderWindowInteractor()

#iren.SetRenderWindow(renWin)
ren.AddActor(cylinderActor)

ren.SetBackground(colors.GetColor3d("BkgColor"))



ren.ResetCamera()

ren.GetActiveCamera().Zoom(1.5)

#renWin.Render() # useless line in kaggle nb
img = vtk_show(ren)

disp.stop()

Image(img)
def vtk_show(renderer, width = default_width, height = default_height, filename = ""):



    renderWindow = vtk.vtkRenderWindow()

    

    renderWindow.SetOffScreenRendering(1)

    renderWindow.AddRenderer(renderer)

    renderWindow.SetSize(width, height)

    renderWindow.Render()

     

    windowToImageFilter = vtk.vtkWindowToImageFilter()

    windowToImageFilter.SetInput(renderWindow)

    windowToImageFilter.Update()

     

    writer = vtk. vtkPNGWriter()

    

    if filename == "":

        writer.SetWriteToMemory(1)

        writer.SetInputConnection(windowToImageFilter.GetOutputPort())

        writer.Write()    

        return bytes(memoryview(writer.GetResult()))

    else:

        writer.SetFileName(filename+".png")

        writer.SetInputConnection(windowToImageFilter.GetOutputPort())

        writer.Write()    

        return None

    

def vtk_render_gif(renderer, N, name, Roll = False, Azimuth = False, Elevation = False, Actor = None, RotateX = False, RotateY = False, RotateZ = False, Zoom = 0, Dolly = 0, standard = True, width = default_width, height = default_height):    

    if standard:

        renderer.ResetCamera()

        camera = renderer.MakeCamera()

        renderer.ResetCameraClippingRange()

        camera.SetPosition(0,0,0)

    os.makedirs(name,exist_ok=True)

    

    if Zoom != 0:

        renderer.GetActiveCamera().Zoom(Zoom)

        

    if Dolly != 0:

        renderer.GetActiveCamera().Dolly(Dolly)

        

    #tmpN = 1

    if N >0: # render gif

        for fi in range(N):

            if Roll:

                renderer.GetActiveCamera().Roll(360//N) 

            if Azimuth:

                renderer.GetActiveCamera().Azimuth(360//N) 

            if Elevation:

                renderer.GetActiveCamera().Elevation(360//N)

            if Actor is not None:

                if RotateX:

                    Actor.RotateX(360//N)

                if RotateY:

                    Actor.RotateY(360//N)

                if RotateZ:

                    Actor.RotateZ(360//N)                    

            vtk_show(renderer,filename = name + "/shot"+str(fi), width = width, height = height)

        # render gif and cleanup

        img_list = []

        for fi in range(N):

            img_list.append(mpimg.imread(name + '/shot' + str(fi) + '.png'))

        shutil.rmtree(name)

        imageio.mimsave(name + ".gif", img_list, duration=0.5)



    #if N == 1: # render png

       #vtk_show(renderer,filename = name + ".gif")

%%time

workdir = '/kaggle/working/vtk_examples/'

name = workdir + 'cylinder'

os.makedirs(workdir,exist_ok=True)

disp = Display().start()



ren = vtk.vtkRenderer()

ren.AddActor(cylinderActor)

ren.SetBackground(colors.GetColor3d("BkgColor"))



vtk_render_gif(ren, N, name, Actor = cylinderActor, RotateX = True, Zoom = 1.5)



disp.stop()



Image(filename=name + ".gif", format='png')
%%time

! wget -q https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Data/kitchen.vtk -O /kaggle/working/vtk_examples/kitchen.vtk

disp = Display().start()

fileName = workdir +'kitchen.vtk'

colors = vtk.vtkNamedColors()

# Set the furniture colors.

colors.SetColor("Furniture", [204, 204, 153, 255])



scalarRange = [0.0, 0.0]

maxTime = 0



aren = vtk.vtkRenderer()



#

# Read the data.

#

reader = vtk.vtkStructuredGridReader()

reader.SetFileName(fileName)

reader.Update()  # Force a read to occur.

reader.GetOutput().GetLength()



if reader.GetOutput().GetPointData().GetScalars():

    reader.GetOutput().GetPointData().GetScalars().GetRange(scalarRange)



if reader.GetOutput().GetPointData().GetVectors():

    maxVelocity = reader.GetOutput().GetPointData().GetVectors().GetMaxNorm()

    maxTime = 4.0 * reader.GetOutput().GetLength() / maxVelocity



#

# Outline around the data.

#

outlineF = vtk.vtkStructuredGridOutlineFilter()

outlineF.SetInputConnection(reader.GetOutputPort())

outlineMapper = vtk.vtkPolyDataMapper()

outlineMapper.SetInputConnection(outlineF.GetOutputPort())

outline = vtk.vtkActor()

outline.SetMapper(outlineMapper)

outline.GetProperty().SetColor(colors.GetColor3d("LampBlack"))



#

# Set up shaded surfaces (i.e., supporting geometry).

#

doorGeom = vtk.vtkStructuredGridGeometryFilter()

doorGeom.SetInputConnection(reader.GetOutputPort())

doorGeom.SetExtent(27, 27, 14, 18, 0, 11)

mapDoor = vtk.vtkPolyDataMapper()

mapDoor.SetInputConnection(doorGeom.GetOutputPort())

mapDoor.ScalarVisibilityOff()

door = vtk.vtkActor()

door.SetMapper(mapDoor)

door.GetProperty().SetColor(colors.GetColor3d("Burlywood"))



window1Geom = vtk.vtkStructuredGridGeometryFilter()

window1Geom.SetInputConnection(reader.GetOutputPort())

window1Geom.SetExtent(0, 0, 9, 18, 6, 12)

mapWindow1 = vtk.vtkPolyDataMapper()

mapWindow1.SetInputConnection(window1Geom.GetOutputPort())

mapWindow1.ScalarVisibilityOff()

window1 = vtk.vtkActor()

window1.SetMapper(mapWindow1)

window1.GetProperty().SetColor(colors.GetColor3d("SkyBlue"))

window1.GetProperty().SetOpacity(.6)



window2Geom = vtk.vtkStructuredGridGeometryFilter()

window2Geom.SetInputConnection(reader.GetOutputPort())

window2Geom.SetExtent(5, 12, 23, 23, 6, 12)

mapWindow2 = vtk.vtkPolyDataMapper()

mapWindow2.SetInputConnection(window2Geom.GetOutputPort())

mapWindow2.ScalarVisibilityOff()

window2 = vtk.vtkActor()

window2.SetMapper(mapWindow2)

window2.GetProperty().SetColor(colors.GetColor3d("SkyBlue"))

window2.GetProperty().SetOpacity(.6)



klower1Geom = vtk.vtkStructuredGridGeometryFilter()

klower1Geom.SetInputConnection(reader.GetOutputPort())

klower1Geom.SetExtent(17, 17, 0, 11, 0, 6)

mapKlower1 = vtk.vtkPolyDataMapper()

mapKlower1.SetInputConnection(klower1Geom.GetOutputPort())

mapKlower1.ScalarVisibilityOff()

klower1 = vtk.vtkActor()

klower1.SetMapper(mapKlower1)

klower1.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower2Geom = vtk.vtkStructuredGridGeometryFilter()

klower2Geom.SetInputConnection(reader.GetOutputPort())

klower2Geom.SetExtent(19, 19, 0, 11, 0, 6)

mapKlower2 = vtk.vtkPolyDataMapper()

mapKlower2.SetInputConnection(klower2Geom.GetOutputPort())

mapKlower2.ScalarVisibilityOff()

klower2 = vtk.vtkActor()

klower2.SetMapper(mapKlower2)

klower2.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower3Geom = vtk.vtkStructuredGridGeometryFilter()

klower3Geom.SetInputConnection(reader.GetOutputPort())

klower3Geom.SetExtent(17, 19, 0, 0, 0, 6)

mapKlower3 = vtk.vtkPolyDataMapper()

mapKlower3.SetInputConnection(klower3Geom.GetOutputPort())

mapKlower3.ScalarVisibilityOff()

klower3 = vtk.vtkActor()

klower3.SetMapper(mapKlower3)

klower3.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower4Geom = vtk.vtkStructuredGridGeometryFilter()

klower4Geom.SetInputConnection(reader.GetOutputPort())

klower4Geom.SetExtent(17, 19, 11, 11, 0, 6)

mapKlower4 = vtk.vtkPolyDataMapper()

mapKlower4.SetInputConnection(klower4Geom.GetOutputPort())

mapKlower4.ScalarVisibilityOff()

klower4 = vtk.vtkActor()

klower4.SetMapper(mapKlower4)

klower4.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower5Geom = vtk.vtkStructuredGridGeometryFilter()

klower5Geom.SetInputConnection(reader.GetOutputPort())

klower5Geom.SetExtent(17, 19, 0, 11, 0, 0)

mapKlower5 = vtk.vtkPolyDataMapper()

mapKlower5.SetInputConnection(klower5Geom.GetOutputPort())

mapKlower5.ScalarVisibilityOff()

klower5 = vtk.vtkActor()

klower5.SetMapper(mapKlower5)

klower5.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower6Geom = vtk.vtkStructuredGridGeometryFilter()

klower6Geom.SetInputConnection(reader.GetOutputPort())

klower6Geom.SetExtent(17, 19, 0, 7, 6, 6)

mapKlower6 = vtk.vtkPolyDataMapper()

mapKlower6.SetInputConnection(klower6Geom.GetOutputPort())

mapKlower6.ScalarVisibilityOff()

klower6 = vtk.vtkActor()

klower6.SetMapper(mapKlower6)

klower6.GetProperty().SetColor(colors.GetColor3d("EggShell"))



klower7Geom = vtk.vtkStructuredGridGeometryFilter()

klower7Geom.SetInputConnection(reader.GetOutputPort())

klower7Geom.SetExtent(17, 19, 9, 11, 6, 6)

mapKlower7 = vtk.vtkPolyDataMapper()

mapKlower7.SetInputConnection(klower7Geom.GetOutputPort())

mapKlower7.ScalarVisibilityOff()

klower7 = vtk.vtkActor()

klower7.SetMapper(mapKlower7)

klower7.GetProperty().SetColor(colors.GetColor3d("EggShell"))



hood1Geom = vtk.vtkStructuredGridGeometryFilter()

hood1Geom.SetInputConnection(reader.GetOutputPort())

hood1Geom.SetExtent(17, 17, 0, 11, 11, 16)

mapHood1 = vtk.vtkPolyDataMapper()

mapHood1.SetInputConnection(hood1Geom.GetOutputPort())

mapHood1.ScalarVisibilityOff()

hood1 = vtk.vtkActor()

hood1.SetMapper(mapHood1)

hood1.GetProperty().SetColor(colors.GetColor3d("Silver"))



hood2Geom = vtk.vtkStructuredGridGeometryFilter()

hood2Geom.SetInputConnection(reader.GetOutputPort())

hood2Geom.SetExtent(19, 19, 0, 11, 11, 16)

mapHood2 = vtk.vtkPolyDataMapper()

mapHood2.SetInputConnection(hood2Geom.GetOutputPort())

mapHood2.ScalarVisibilityOff()

hood2 = vtk.vtkActor()

hood2.SetMapper(mapHood2)

hood2.GetProperty().SetColor(colors.GetColor3d("Furniture"))



hood3Geom = vtk.vtkStructuredGridGeometryFilter()

hood3Geom.SetInputConnection(reader.GetOutputPort())

hood3Geom.SetExtent(17, 19, 0, 0, 11, 16)

mapHood3 = vtk.vtkPolyDataMapper()

mapHood3.SetInputConnection(hood3Geom.GetOutputPort())

mapHood3.ScalarVisibilityOff()

hood3 = vtk.vtkActor()

hood3.SetMapper(mapHood3)

hood3.GetProperty().SetColor(colors.GetColor3d("Furniture"))



hood4Geom = vtk.vtkStructuredGridGeometryFilter()

hood4Geom.SetInputConnection(reader.GetOutputPort())

hood4Geom.SetExtent(17, 19, 11, 11, 11, 16)

mapHood4 = vtk.vtkPolyDataMapper()

mapHood4.SetInputConnection(hood4Geom.GetOutputPort())

mapHood4.ScalarVisibilityOff()

hood4 = vtk.vtkActor()

hood4.SetMapper(mapHood4)

hood4.GetProperty().SetColor(colors.GetColor3d("Furniture"))



hood6Geom = vtk.vtkStructuredGridGeometryFilter()

hood6Geom.SetInputConnection(reader.GetOutputPort())

hood6Geom.SetExtent(17, 19, 0, 11, 16, 16)

mapHood6 = vtk.vtkPolyDataMapper()

mapHood6.SetInputConnection(hood6Geom.GetOutputPort())

mapHood6.ScalarVisibilityOff()

hood6 = vtk.vtkActor()

hood6.SetMapper(mapHood6)

hood6.GetProperty().SetColor(colors.GetColor3d("Furniture"))



cookingPlateGeom = vtk.vtkStructuredGridGeometryFilter()

cookingPlateGeom.SetInputConnection(reader.GetOutputPort())

cookingPlateGeom.SetExtent(17, 19, 7, 9, 6, 6)

mapCookingPlate = vtk.vtkPolyDataMapper()

mapCookingPlate.SetInputConnection(cookingPlateGeom.GetOutputPort())

mapCookingPlate.ScalarVisibilityOff()

cookingPlate = vtk.vtkActor()

cookingPlate.SetMapper(mapCookingPlate)

cookingPlate.GetProperty().SetColor(colors.GetColor3d("Tomato"))



filterGeom = vtk.vtkStructuredGridGeometryFilter()

filterGeom.SetInputConnection(reader.GetOutputPort())

filterGeom.SetExtent(17, 19, 7, 9, 11, 11)

mapFilter = vtk.vtkPolyDataMapper()

mapFilter.SetInputConnection(filterGeom.GetOutputPort())

mapFilter.ScalarVisibilityOff()

sgfilter = vtk.vtkActor()

sgfilter.SetMapper(mapFilter)

sgfilter.GetProperty().SetColor(colors.GetColor3d("Furniture"))

#

# regular streamlines

#

line = vtk.vtkLineSource()

line.SetResolution(39)

line.SetPoint1(0.08, 2.50, 0.71)

line.SetPoint2(0.08, 4.50, 0.71)

rakeMapper = vtk.vtkPolyDataMapper()

rakeMapper.SetInputConnection(line.GetOutputPort())

rake = vtk.vtkActor()

rake.SetMapper(rakeMapper)



streamers = vtk.vtkStreamTracer()

# streamers.DebugOn()

streamers.SetInputConnection(reader.GetOutputPort())

streamers.SetSourceConnection(line.GetOutputPort())

streamers.SetMaximumPropagation(maxTime)

streamers.SetInitialIntegrationStep(.5)

streamers.SetMinimumIntegrationStep(.1)

streamers.SetIntegratorType(2)

streamers.Update()



streamersMapper = vtk.vtkPolyDataMapper()

streamersMapper.SetInputConnection(streamers.GetOutputPort())

streamersMapper.SetScalarRange(scalarRange)



lines = vtk.vtkActor()

lines.SetMapper(streamersMapper)

lines.GetProperty().SetColor(colors.GetColor3d("Black"))



aren.TwoSidedLightingOn()



aren.AddActor(outline)

aren.AddActor(door)

aren.AddActor(window1)

aren.AddActor(window2)

aren.AddActor(klower1)

aren.AddActor(klower2)

aren.AddActor(klower3)

aren.AddActor(klower4)

aren.AddActor(klower5)

aren.AddActor(klower6)

aren.AddActor(klower7)

aren.AddActor(hood1)

aren.AddActor(hood2)

aren.AddActor(hood3)

aren.AddActor(hood4)

aren.AddActor(hood6)

aren.AddActor(cookingPlate)

aren.AddActor(sgfilter)

aren.AddActor(lines)

aren.AddActor(rake)



aren.SetBackground(colors.GetColor3d("SlateGray"))



aCamera = vtk.vtkCamera()

aren.SetActiveCamera(aCamera)

aren.ResetCamera()



aCamera.SetFocalPoint(3.505, 2.505, 1.255)

aCamera.SetPosition(3.505, 24.6196, 1.255)

aCamera.SetViewUp(0, 0, 1)

aCamera.Azimuth(60)

aCamera.Elevation(30)

aren.ResetCameraClippingRange()



name = workdir + 'kitchen'

vtk_render_gif(aren, N, name, Azimuth = True, Dolly = 1.5)

disp.stop()



Image(filename=name + ".gif", format='png')    
! wget -q https://raw.githubusercontent.com/lorensen/VTKWikiExamples/master/Testing/Data/frogtissue.mhd -P /kaggle/working/vtk_examples 

! wget -q https://raw.githubusercontent.com/lorensen/VTKWikiExamples/master/Testing/Data/frogtissue.zraw  -P /kaggle/working/vtk_examples

def CreateFrogLut():

    colors = vtk.vtkNamedColors()



    colorLut = vtk.vtkLookupTable()

    colorLut.SetNumberOfColors(17)

    colorLut.SetTableRange(0, 16)

    colorLut.Build()



    colorLut.SetTableValue(0, 0, 0, 0, 0)

    colorLut.SetTableValue(1, colors.GetColor4d("salmon"))  # blood

    colorLut.SetTableValue(2, colors.GetColor4d("beige"))  # brain

    colorLut.SetTableValue(3, colors.GetColor4d("orange"))  # duodenum

    colorLut.SetTableValue(4, colors.GetColor4d("misty_rose"))  # eye_retina

    colorLut.SetTableValue(5, colors.GetColor4d("white"))  # eye_white

    colorLut.SetTableValue(6, colors.GetColor4d("tomato"))  # heart

    colorLut.SetTableValue(7, colors.GetColor4d("raspberry"))  # ileum

    colorLut.SetTableValue(8, colors.GetColor4d("banana"))  # kidney

    colorLut.SetTableValue(9, colors.GetColor4d("peru"))  # l_intestine

    colorLut.SetTableValue(10, colors.GetColor4d("pink"))  # liver

    colorLut.SetTableValue(11, colors.GetColor4d("powder_blue"))  # lung

    colorLut.SetTableValue(12, colors.GetColor4d("carrot"))  # nerve

    colorLut.SetTableValue(13, colors.GetColor4d("wheat"))  # skeleton

    colorLut.SetTableValue(14, colors.GetColor4d("violet"))  # spleen

    colorLut.SetTableValue(15, colors.GetColor4d("plum"))  # stomach



    return colorLut



def CreateTissueMap():

    tissueMap = dict()

    tissueMap["blood"] = 1

    tissueMap["brain"] = 2

    tissueMap["duodenum"] = 3

    tissueMap["eyeRetina"] = 4

    tissueMap["eyeWhite"] = 5

    tissueMap["heart"] = 6

    tissueMap["ileum"] = 7

    tissueMap["kidney"] = 8

    tissueMap["intestine"] = 9

    tissueMap["liver"] = 10

    tissueMap["lung"] = 11

    tissueMap["nerve"] = 12

    tissueMap["skeleton"] = 13

    tissueMap["spleen"] = 14

    tissueMap["stomach"] = 15



    return tissueMap



def CreateFrogActor(fileName, tissue):

    reader = vtk.vtkMetaImageReader()

    reader.SetFileName(fileName)

    reader.Update()



    selectTissue = vtk.vtkImageThreshold()

    selectTissue.ThresholdBetween(tissue, tissue)

    selectTissue.SetInValue(255)

    selectTissue.SetOutValue(0)

    selectTissue.SetInputConnection(reader.GetOutputPort())



    gaussianRadius = 1

    gaussianStandardDeviation = 2.0

    gaussian = vtk.vtkImageGaussianSmooth()

    gaussian.SetStandardDeviations(gaussianStandardDeviation, gaussianStandardDeviation, gaussianStandardDeviation)

    gaussian.SetRadiusFactors(gaussianRadius, gaussianRadius, gaussianRadius)

    gaussian.SetInputConnection(selectTissue.GetOutputPort())



    isoValue = 127.5

    mcubes = vtk.vtkMarchingCubes()

    mcubes.SetInputConnection(gaussian.GetOutputPort())

    mcubes.ComputeScalarsOff()

    mcubes.ComputeGradientsOff()

    mcubes.ComputeNormalsOff()

    mcubes.SetValue(0, isoValue)



    smoothingIterations = 5

    passBand = 0.001

    featureAngle = 60.0

    smoother = vtk.vtkWindowedSincPolyDataFilter()

    smoother.SetInputConnection(mcubes.GetOutputPort())

    smoother.SetNumberOfIterations(smoothingIterations)

    smoother.BoundarySmoothingOff()

    smoother.FeatureEdgeSmoothingOff()

    smoother.SetFeatureAngle(featureAngle)

    smoother.SetPassBand(passBand)

    smoother.NonManifoldSmoothingOn()

    smoother.NormalizeCoordinatesOn()

    smoother.Update()



    normals = vtk.vtkPolyDataNormals()

    normals.SetInputConnection(smoother.GetOutputPort())

    normals.SetFeatureAngle(featureAngle)



    stripper = vtk.vtkStripper()

    stripper.SetInputConnection(normals.GetOutputPort())



    mapper = vtk.vtkPolyDataMapper()

    mapper.SetInputConnection(stripper.GetOutputPort())



    actor = vtk.vtkActor()

    actor.SetMapper(mapper)



    return actor
%%time

disp = Display().start()



fileName = workdir + 'frogtissue.mhd'

tissueMap = CreateTissueMap()



colors = vtk.vtkNamedColors()



colorLut = CreateFrogLut()

renderer = vtk.vtkRenderer()



for tissue in [t for t in tissueMap]:

    actor = CreateFrogActor(fileName, tissueMap[tissue])

    actor.GetProperty().SetDiffuseColor( colorLut.GetTableValue(tissueMap[tissue])[:3])

    actor.GetProperty().SetSpecular(.5)

    actor.GetProperty().SetSpecularPower(10)

    renderer.AddActor(actor)



renderer.GetActiveCamera().SetViewUp(0, 0, -1)

renderer.GetActiveCamera().SetPosition(0, -1, 0)



renderer.GetActiveCamera().Azimuth(210)

renderer.GetActiveCamera().Elevation(30)

renderer.ResetCamera()



name = workdir + 'frog'

vtk_render_gif(renderer, N, name, Azimuth = True, Dolly = 1.5)

disp.stop()



Image(filename=name + ".gif", format='png')    
! wget https://download.slicer.org/bitstream/1023242 -O Slicer-4.3.0-linux-amd64.tar.gz -q >>/dev/null

! tar xzf Slicer-4.3.0-linux-amd64.tar.gz -C ~/ >>/dev/null

! apt-get install libglu1 -qq >>/dev/null

! apt-get install libpulse-mainloop-glib0 -qq >>/dev/null

! apt-get install libegl-mesa0 -y libegl1 -qq >>/dev/null
import os

filepath = ''

def MakeFile(file_name):

    temp_path = filepath + file_name

    with open(file_name, 'w') as f:

        f.write('''\

# use a slicer scripted module logic

from SampleData import SampleDataLogic

SampleDataLogic().downloadMRHead()

head = slicer.util.getNode("MRHead")



# use a vtk class

threshold = vtk.vtkImageThreshold()

threshold.SetInputData(head.GetImageData())

threshold.ThresholdBetween(100, 200)

threshold.SetInValue(255)

threshold.SetOutValue(0)



#  use a slicer-specific C++ class

erode = slicer.vtkImageErode()

erode.SetInputConnection(threshold.GetOutputPort())

erode.SetNeighborTo4()  

erode.Update()          



head.SetAndObserveImageData(erode.GetOutputDataObject(0))



slicer.util.saveNode(head, "/kaggle/working/eroded.nrrd")



import ScreenCapture

l=ScreenCapture.ScreenCaptureLogic()

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeRed')), '/kaggle/working/red.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeGreen')), '/kaggle/working/green.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeYellow')), '/kaggle/working/yellow.png')

exit()      

''')

MakeFile('slicer_code.py')



! xvfb-run -a ~/Slicer-4.10.2-linux-amd64/Slicer --no-splash --python-script slicer_code.py  > /dev/null
plt.figure(figsize=(20,10))

imgs = ['red','green','yellow']

for fi in range(len(imgs)):

    plt.subplot(1,3,fi+1) 

    im = mpimg.imread(imgs[fi]+'.png')

    plt.imshow(im) 
def MakeFile(file_name):

    temp_path = filepath + file_name

    with open(file_name, 'w') as f:

        f.write('''\

# use a slicer scripted module logic

from SampleData import SampleDataLogic

SampleDataLogic().downloadCTChest()

head = slicer.util.getNode("CTChest")



# use a vtk class

threshold = vtk.vtkImageThreshold()

threshold.SetInputData(head.GetImageData())

threshold.ThresholdBetween(100, 800)

threshold.SetInValue(255)

threshold.SetOutValue(0)



#  use a slicer-specific C++ class

erode = slicer.vtkImageErode()

erode.SetInputConnection(threshold.GetOutputPort())

erode.SetNeighborTo4()  

erode.Update()          



head.SetAndObserveImageData(erode.GetOutputDataObject(0))



slicer.util.saveNode(head, "/kaggle/working/eroded.nrrd")



import ScreenCapture

l=ScreenCapture.ScreenCaptureLogic()

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeRed')), '/kaggle/working/red.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeGreen')), '/kaggle/working/green.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeYellow')), '/kaggle/working/yellow.png')



renderer = slicer.app.layoutManager().threeDWidget(0).threeDView()

#centerViewport = [0.33, 0.0, .66, 1.0]

#renderer.SetViewport(centerViewport)



#width = 1000

#height = 1000

renderWindow = renderer.renderWindow()

#renderWindow.SetSize(width, height)

renderWindow.SetAlphaBitPlanes(1)

wti = vtk.vtkWindowToImageFilter()

wti.SetInputBufferTypeToRGBA()

wti.SetInput(renderWindow)

writer = vtk.vtkPNGWriter()

writer.SetFileName("screenshot.png")

writer.SetInputConnection(wti.GetOutputPort())

writer.Write()



exit()      

''')

MakeFile('slicer_code.py')



# https://discourse.slicer.org/t/running-slicer-without-gui/11720/4

! xvfb-run -a ~/Slicer-4.10.2-linux-amd64/Slicer --no-splash --python-script slicer_code.py  > /dev/null
plt.figure(figsize=(20,10))



imgs = ['red','green','yellow']

for fi in range(len(imgs)):

    plt.subplot(1,4,fi+1) 

    im = mpimg.imread(imgs[fi]+'.png')

    plt.imshow(im) 
plt.figure(figsize=(5,5))

dir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/'

import pydicom as dcm

plt.imshow(dcm.dcmread(dir + os.listdir(dir)[0]).pixel_array)
def MakeFile(file_name):

    temp_path = filepath + file_name

    with open(file_name, 'w') as f:

        f.write('''\

# use a slicer scripted module logic

from SampleData import SampleDataLogic

#SampleDataLogic().downloadCTChest()

#head = slicer.util.getNode("CTChest")



from DICOMLib import DICOMUtils

dicomDataDir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/' # input folder with DICOM files

loadedNodeIDs = []  # this list will contain the list of all loaded node IDs



db=slicer.dicomDatabase



with DICOMUtils.TemporaryDICOMDatabase() as db:

    DICOMUtils.importDicom(dicomDataDir, db)

    patientUIDs = db.patients()

    for patientUID in patientUIDs:

        #loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))

        loadedNodeIDs.append(DICOMUtils.loadPatientByUID(patientUID))



head = slicer.util.getNode()

# use a vtk class

threshold = vtk.vtkImageThreshold()

#threshold.SetInputData(head.GetImageData())

#threshold.ThresholdBetween(0, 500)

#threshold.SetInValue(255)

#threshold.SetOutValue(0)



import ScreenCapture

l=ScreenCapture.ScreenCaptureLogic()

node_red = slicer.util.getNode('vtkMRMLSliceNodeRed')

l.captureImageFromView(l.viewFromNode(node_red), '/kaggle/working/red.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeGreen')), '/kaggle/working/green.png')

l.captureImageFromView(l.viewFromNode(slicer.util.getNode('vtkMRMLSliceNodeYellow')), '/kaggle/working/yellow.png')





slicer.util.resetSliceViews()



layoutManager = slicer.app.layoutManager()

for sliceViewName in layoutManager.sliceViewNames():

  controller = layoutManager.sliceWidget(sliceViewName).sliceController()

  controller.setSliceVisible(True)

  

threeDWidget = layoutManager.threeDWidget(0)

threeDView = threeDWidget.threeDView()

threeDView.resetFocalPoint()



renderWindow = threeDView.renderWindow()

renderWindow.SetAlphaBitPlanes(1)

wti = vtk.vtkWindowToImageFilter()

wti.SetInputBufferTypeToRGBA()

wti.SetInput(renderWindow)

writer = vtk.vtkPNGWriter()

writer.SetFileName("screenshot.png")

writer.SetInputConnection(wti.GetOutputPort())

writer.Write()



#layoutManager = slicer.app.layoutManager()

#threeDWidget = layoutManager.threeDWidget(0)

#threeDView = threeDWidget.threeDView()

#threeDView.resetFocalPoint()



exit()      

''')

MakeFile('slicer_code.py')



# https://discourse.slicer.org/t/running-slicer-without-gui/11720/4

! xvfb-run -a ~/Slicer-4.10.2-linux-amd64/Slicer --no-splash --python-script slicer_code.py > /dev/null
plt.figure(figsize=(20,10))



imgs = ['red','green','yellow']

for fi in range(len(imgs)):

    plt.subplot(1,4,fi+1) 

    im = mpimg.imread(imgs[fi]+'.png')

    plt.imshow(im) 

! rm Slicer-4.3.0-linux-amd64.tar.gz

! rm slicer_code.py