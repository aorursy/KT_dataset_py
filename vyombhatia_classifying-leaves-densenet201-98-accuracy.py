from fastai.vision.all import *              # The asterisk '*' here mean everything.
from fastai.metrics import accuracy          # Using accuracy as metric for our model.
path = "../input/cotton-disease-dataset/Cotton Disease"

data = ImageDataLoaders.from_folder(path,                              # Path refers to where the folder containing the files is. 
                                    
                                    valid_pct = 0.3,                   # This refers to the ratio of validation data the model would test itself on.          
                                   
                                    bs = 4,                            # bs refers to Batch Size.
                                    
                                    shuffle_train = True,              # This shuffles the data before grabbing batches.
                                   
                                    item_tfms = Resize(512),           # item_tfms transforms the images to to the given specifications before 
                                                                       #  grabbing batches.
                                   
                                    batch_tfms = aug_transforms(       # batch_tfms transforms the batches.
                                        size = 256, min_scale = 0.75)) 
data.show_batch(nrows = 1, ncols = 4)
learn = cnn_learner(data,                          # data refers to the data preprocessed by the DataLoader
                    
                    densenet201,                   # Since we are using Transfer Learning here, lets import the DenseNet201 
                    
                    metrics = accuracy)         
# Training the model for 5 epochs:
learn.fine_tune(5)
# Lets see where our classifier got confused:

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize = (12,12), dpi = 60)
# Lets predict on a new images that the model has not seen:

img = plt.imread("../input/cotton-disease-dataset/Cotton Disease/test/diseased cotton leaf/dis_leaf (124).jpg")

classofimg, idx, probability = learn.predict(img)

print("Image is of a",classofimg + ".")\

plt.imshow(img)