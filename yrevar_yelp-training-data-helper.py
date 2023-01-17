import os
import numpy as np
import pandas as pd
import IPython
from IPython.display import SVG

import matplotlib.pyplot as plt
from scipy import misc

%matplotlib inline
plt.rcParams['figure.figsize'] = (15.0, 15.0) # set default size of plots
# default image preprocess
def dflt_img_preprocess_fn(img, resize_dim=(480, 480, 3)):
    return misc.imresize(img, resize_dim, interp="nearest")
                
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# prepare image for vggnet
def prepare_image_for_vggnet(img, resize_dim=(224, 224, 3), sub_vgg_mean=False, depth_first=True, dtype=np.float32):

    if resize_dim == -1:
        img = img.astype(dtype)
    else:
        img = misc.imresize(img, resize_dim, interp="nearest").astype(dtype)

    if sub_vgg_mean:
        img[:,:,0] -= dtype(103.939)
        img[:,:,1] -= dtype(116.779)
        img[:,:,2] -= dtype(123.68)

    if depth_first:
        img = img.transpose((2,0,1))

    return img

"""
train_photo_biz_tbl: training dataframe, photo - biz id 
train_biz_attrib_tbl: training dataframe, biz id - attributes
test_photo_biz_tbl: test dataframe, photo - biz id
"""
class YelpDataCrawler(object):
    
    restaurant_attributes = ["good_for_lunch", "good_for_dinner", "takes_reservations", 
                             "outdoor_seating", "restaurant_is_expensive", "has_alcohol", 
                             "has_table_service", "ambience_is_classy", "good_for_kids"]
   
    def __init__(self, save_processed_data_at="../pdata"):
        
        self.save_processed = save_processed_data_at != None
        self.processed_data_dir = save_processed_data_at
        
    def load_train_data(self, data_dir="../input", 
                                        images_dir="train_photos",
                                        photos_biz_tbl="train_photo_to_biz_ids.csv", 
                                        biz_attrib_tbl="train.csv"):
        
        self.data_dir = data_dir
        self.train_images_dir = os.path.join(data_dir, images_dir)
        train_pdata_dir = os.path.join(self.processed_data_dir,'train') if self.save_processed else ""
        
        if self.save_processed and os.path.isdir(train_pdata_dir):
            
            self.train_photo_biz_tbl = pd.read_csv(os.path.join(train_pdata_dir, photos_biz_tbl))
            self.train_biz_attrib_tbl = pd.read_csv(os.path.join(train_pdata_dir, biz_attrib_tbl))
        else:
            
            self.train_photo_biz_tbl = pd.read_csv(os.path.join(data_dir, photos_biz_tbl))
            self.train_biz_attrib_tbl = pd.read_csv(os.path.join(data_dir, biz_attrib_tbl))
            
            # remove nas and duplicates
            self.train_biz_attrib_tbl = self.train_biz_attrib_tbl.dropna()
            self.train_biz_attrib_tbl = self.train_biz_attrib_tbl.drop_duplicates()
            
            # remove entries with no attributes
            self.train_photo_biz_tbl = self.train_photo_biz_tbl[self.train_photo_biz_tbl['business_id'].apply(
                                                        lambda x: x in self.train_biz_attrib_tbl.business_id.values)]
            
            if self.save_processed:
                
                # create dir to store processed data 
                mkdir_p(os.path.join(train_pdata_dir))

                # save processed tables to disk
                self.train_photo_biz_tbl.to_csv(os.path.join(train_pdata_dir, photos_biz_tbl), index=False)
                self.train_biz_attrib_tbl.to_csv(os.path.join(train_pdata_dir, biz_attrib_tbl), index=False)
                
        self.num_train_images = self.train_photo_biz_tbl.photo_id.size
        self.num_train_biz = self.train_photo_biz_tbl.business_id.unique().size
        self.num_out_classes = len(self.restaurant_attributes)
        
    def load_test_data(self, test_data_dir="../input",
                                        images_dir="test_photos",
                                        photos_biz_tbl="test_photo_to_biz.csv"):
        
        self.test_data_dir = test_data_dir
        self.test_images_dir = os.path.join(test_data_dir, images_dir)
        test_pdata_dir = os.path.join(self.processed_data_dir,'test') if self.save_processed else ""
    
        if self.save_processed and os.path.isdir(test_pdata_dir):
            
            self.test_photo_biz_tbl = pd.read_csv(os.path.join(test_pdata_dir, photos_biz_tbl))
        else:
            
            self.test_photo_biz_tbl = pd.read_csv(os.path.join(test_data_dir, photos_biz_tbl))
            
            if self.save_processed:
                
                # create dir to store processed data 
                mkdir_p(os.path.join(test_pdata_dir))

                # save processed tables to disk
                self.test_photo_biz_tbl.to_csv(os.path.join(test_pdata_dir, photos_biz_tbl), index=False)
                
        self.num_test_images = self.test_photo_biz_tbl.photo_id.size
        self.num_test_biz = self.test_photo_biz_tbl.business_id.unique().size
    
    # get business attributes by biz id
    def get_train_biz_attribs_by_biz_id(self, biz_id):
        return self.train_biz_attrib_tbl[
                        self.train_biz_attrib_tbl.business_id == biz_id]['labels'].as_matrix()[0].split()
    
    # get one-hot encoded business attributes by biz id
    def get_train_biz_attribs_one_hot_by_biz_id(self, biz_id):
        return np.array([(0,1)[str(i) in self.get_train_biz_attribs_by_biz_id(biz_id)] \
                                                            for i in range(len(self.restaurant_attributes))])
    
    # read image from disk
    def read_image(self, img_file_path):
        return misc.imread(img_file_path)
    
    # get training image by id
    def get_train_image_by_img_id(self, img_id):
        return self.read_image(os.path.join(self.train_images_dir, str(img_id)+".jpg"))
    
    # get training image by index (image to biz table)
    def get_train_image_by_img_idx(self, img_idx):
        return self.get_train_image_by_img_id(self.train_photo_biz_tbl.iloc[img_idx]["photo_id"])
    
    # get training image ids associated with a biz id
    def get_train_image_ids_by_biz_id(self, biz_id):
        return self.train_photo_biz_tbl[self.train_photo_biz_tbl.business_id == biz_id]["photo_id"].as_matrix()
    
    # get training images by biz id
    def get_train_images_by_biz_id(self, biz_id, img_preprocess_fn=dflt_img_preprocess_fn, 
                                                                    req_max_imgs=10, shuffle=False):
        
        biz_img_ids = self.get_train_image_ids_by_biz_id(biz_id)[:req_max_imgs]
        if shuffle: 
            np.random.shuffle(biz_img_ids)
        return np.array([img_preprocess_fn(self.get_train_image_by_img_id(img_id)) for img_id in biz_img_ids])

    # read test image by id
    def get_test_image_by_img_id(self, img_id):
        return self.read_image(os.path.join(self.test_images_dir, str(img_id)+".jpg"))    
    
    # get test image by index (image to biz table)
    def get_test_image_by_img_idx(self, img_idx):
        return self.get_test_image_by_img_id(self.test_photo_biz_tbl.iloc[img_idx]["photo_id"])

    # get test image ids associated with a biz id
    def get_test_image_ids_by_biz_id(self, biz_id):
        return self.test_photo_biz_tbl[self.test_photo_biz_tbl.business_id == biz_id]["photo_id"].as_matrix()
    
    # get test images by biz id
    def get_test_images_by_biz_id(self, biz_id, img_preprocess_fn=dflt_img_preprocess_fn,
                                                                      req_max_imgs=10, shuffle=False):
        
        biz_img_ids = self.get_test_image_ids_by_biz_id(biz_id)[:req_max_imgs]
        if shuffle: 
            np.random.shuffle(biz_img_ids)
        return np.array([img_preprocess_fn(self.get_test_image_by_img_id(img_id)) for img_id in biz_img_ids])
    
    def sample_train_images(self, sample_size, img_preprocess_fn=dflt_img_preprocess_fn, 
                                                                        replace=False, rand_seed=None):

        shuffle = True
        
        if self.train_photo_biz_tbl.index.size < sample_size:
            raise Exception("TrainSampleFinished")
        
        sample_photobiz_tbl = self.train_photo_biz_tbl.sample(n=sample_size, replace=replace, random_state=rand_seed) 
        
        X_inputs = np.array([img_preprocess_fn(self.get_train_image_by_img_id(img_id)) \
                                                for img_id in sample_photobiz_tbl["photo_id"] ])
        y_labels = np.array([self.get_train_biz_attribs_one_hot_by_biz_id(biz_id) \
                                                for biz_id in sample_photobiz_tbl["business_id"]])
        
        if replace == False:
            self.train_photo_biz_tbl = self.train_photo_biz_tbl.drop(sample_photobiz_tbl.index)            
            
        return X_inputs, y_labels
    
    def iterate_train_images_by_class(self, class_name, req_max_biz=10, req_max_imgs=10, shuffle=False):
        
        bizattrib_tbl = self.train_biz_attrib_tbl
        photobiz_tbl = self.train_photo_biz_tbl
        filtered_tbl = bizattrib_tbl[bizattrib_tbl["labels"].apply(
                                lambda x: str(self.restaurant_attributes.index(class_name)) in str(x).split())]
        
        if shuffle:
            filtered_tbl = filtered_tbl.reindex(np.random.permutation(filtered_tbl.index))
        
        for biz_id in filtered_tbl[:req_max_biz]["business_id"]:
            for img_id in photobiz_tbl[photobiz_tbl.business_id == biz_id][:req_max_imgs]["photo_id"]:
                yield self.get_train_image_by_img_id(img_id)
    
    def iterate_train_biz_images(self, img_preprocess_fn=dflt_img_preprocess_fn, req_max_imgs=10, shuffle=False):
            
        for biz_id in self.train_photo_biz_tbl.business_id.unique():
            biz_imgs = self.get_train_images_by_biz_id(biz_id, img_preprocess_fn, req_max_imgs, shuffle)
            yield biz_id, biz_imgs
            
    def iterate_test_biz_images(self, img_preprocess_fn=dflt_img_preprocess_fn, req_max_imgs=10, shuffle=False):

        for biz_id in self.test_photo_biz_tbl.business_id.unique():
            
            biz_imgs = self.get_test_images_by_biz_id(biz_id, img_preprocess_fn, req_max_imgs, shuffle)
            yield biz_id, biz_imgs
# To save loading time, use 'save_processed_data_at = {your preferred dir or use default}'
YelpData = YelpDataCrawler(save_processed_data_at="None")
YelpData.load_train_data()
num_samples = 16
xtr, ytr = YelpData.sample_train_images(num_samples, img_preprocess_fn=dflt_img_preprocess_fn, replace=False)
for i in range(num_samples):
    plt.subplot(np.sqrt(num_samples)+1, np.sqrt(num_samples), i+1)
    title_str = "".join([ YelpData.restaurant_attributes[idx] + "\n" for idx,val in enumerate(ytr[i]) if val])
    plt.title(title_str)
    plt.imshow(xtr[i])
    plt.axis('off')
plt.tight_layout(pad=0)
class_names =  YelpData.restaurant_attributes
req_max_biz, req_max_imgs_per_biz = 3, 3

for i, class_name in enumerate(class_names):
    fig = plt.figure(i+1, figsize=(10,10))
    st = fig.suptitle(class_name + " (raw images)", fontsize="x-large")    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    for image_no, img in enumerate(YelpData.iterate_train_images_by_class(
                                       class_name, req_max_biz, req_max_imgs_per_biz, shuffle=True)):
        fig.add_subplot(req_max_biz, req_max_imgs_per_biz, image_no+1)
        plt.axis("off")
        plt.imshow(img)
biz_id = YelpData.train_photo_biz_tbl.business_id[np.random.randint(0, YelpData.num_train_biz)]
X_train = YelpData.get_train_images_by_biz_id(biz_id, req_max_imgs=36)
y_train = YelpData.get_train_biz_attribs_by_biz_id(biz_id)
num_imgs = len(X_train)

fig = plt.figure(1, figsize=(15,15))
fig.subplots_adjust(wspace=0.01, hspace=0.01)
title_str = "biz_id = " + str(biz_id) + "\n" \
            + "".join([ YelpData.restaurant_attributes[idx] + "\n" for idx,val in enumerate(y_train) if val])
plt.suptitle(title_str, fontsize="x-large")
for i in range(num_imgs):
    plt.subplot(np.ceil(np.sqrt(num_imgs)), np.ceil(np.sqrt(num_imgs)), i+1)
    plt.imshow(X_train[i])
    plt.axis("off")
def image_proprocess_vggnet(img):
    img = prepare_image_for_vggnet(img, resize_dim=(224,224,3), 
                                            sub_vgg_mean=True, depth_first=False, dtype=np.float32)
    return img

max_epoch = 1
num_epoch = 0
batch_size = 16
batch_count = 0
while num_epoch < max_epoch:
    try:
        batch_count += 1
        print("Training batch {}/{}".format(batch_count, int(YelpData.num_train_images/batch_size)))
        xtr, ytr = YelpData.sample_train_images(batch_size, img_preprocess_fn=image_proprocess_vggnet, replace=False)
        """
        TRAIN YOUR MODEL HERE
        """
        fig = plt.figure(i+1, figsize=(15,15))
        st = fig.suptitle("Training Input Sample", fontsize="x-large")
        # e.g. code
        for i in range(batch_size):
            plt.subplot(np.sqrt(batch_size)+1, np.sqrt(batch_size), i+1)
            plt.axis('off')
            title_str = "".join([ YelpData.restaurant_attributes[idx] + "\n" \
                                             for idx,val in enumerate(ytr[i]) if val])
            plt.title(title_str)
            plt.imshow(xtr[i])
        plt.tight_layout(pad=0)
        break
        

    except Exception as train_except:
        if "TrainSampleFinished" == train_except.args[0]:
            num_epoch += 1
            print("Train Epoch {}/{}".format(num_epoch, max_epoch))
            # reload training data
            YelpData.load_train_data()
            continue
print("Finished training...")
# Tile business images to feed into network

num_biz_sampled = 0
train_biz_ids = YelpData.train_biz_attrib_tbl.business_id.unique()

# reset train parameters
def restart_train_minibatch_biz_indexed(start_from_biz_no=0):
    global num_biz_sampled 
    num_biz_sampled = start_from_biz_no

num_biz_image_tiles = 9 # must be a perfect square
W, H, C = 224, 224, 3

def image_proprocess_vggnet(img):
    img = prepare_image_for_vggnet(img, resize_dim=(W,H,C), sub_vgg_mean=True, depth_first=True, dtype=np.float32)
    return img

def get_train_minibatch_biz_indexed(num_biz):
    
    global num_biz_sampled
    batch_indx = 0
    W_tile_cnt = H_tile_cnt = int(np.sqrt(num_biz_image_tiles))
    WT, HT = W_tile_cnt*W, H_tile_cnt*H
    
    Xtr_tiled = np.zeros((num_biz, 3, WT, HT))
    ytr_tiled = np.zeros((num_biz, num_biz_image_tiles))
    
    for biz_id in train_biz_ids[num_biz_sampled: num_biz_sampled+num_biz]:
        
        # discard business with insufficient images
        biz_img_count = YelpData.get_train_image_ids_by_biz_id(biz_id).size
        if biz_img_count < num_biz_image_tiles:
            print("Discarding business {} (insufficient no. of images {}, required )").format(
                                                                biz_id, biz_img_count, num_biz_image_tiles)
            continue
            
        xtr = YelpData.get_train_images_by_biz_id(
                        biz_id, image_proprocess_vggnet, req_max_imgs=num_biz_image_tiles, shuffle=True)
        
        Xtr_tiled[batch_indx, :, :, :] = xtr.transpose(1,0,2,3).reshape(
                        C,W_tile_cnt,H_tile_cnt,W,H).transpose(0,1,3,2,4).reshape(C, WT, HT)
        ytr_tiled[batch_indx,:] = YelpData.get_train_biz_attribs_one_hot_by_biz_id(biz_id)
        
        num_biz_sampled += 1
        batch_indx += 1
    return Xtr_tiled, ytr_tiled