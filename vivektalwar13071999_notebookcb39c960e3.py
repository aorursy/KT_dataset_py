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

