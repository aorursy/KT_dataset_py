from scipy.io import loadmat

from PIL import Image

import numpy as np

import os

from glob import glob

import cv2

import argparse



# if the height or width of the image are not within the min_size and max_size

# then get the resized height , width and its magnification

def cal_new_size(im_h, im_w, min_size, max_size):

    if im_h < im_w:

        if im_h < min_size:

            ratio = 1.0 * min_size / im_h

            im_h = min_size

            im_w = round(im_w*ratio)

        elif im_h > max_size:

            ratio = 1.0 * max_size / im_h

            im_h = max_size

            im_w = round(im_w*ratio)

        else:

            ratio = 1.0

    else:

        if im_w < min_size:

            ratio = 1.0 * min_size / im_w

            im_w = min_size

            im_h = round(im_h*ratio)

        elif im_w > max_size:

            ratio = 1.0 * max_size / im_w

            im_w = max_size

            im_h = round(im_h*ratio)

        else:

            ratio = 1.0

    return im_h, im_w, ratio





# find the distance of the nearest point

# we treat each point as a vector from the origin to that point

# distance = sqrt(|vector A|**2 - 2 * (vector A) * (vector B) + |vector B|**2)

def find_dis(point, im_path):

    square = np.sum(point*point, axis=1)



    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))

    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)

    return dis



# return "Image RGB Array" and "Head Points Array"

def generate_data(im_path):

    loc, fn = os.path.split(im_path)

    im = Image.open(im_path)

    im_w, im_h = im.size

    mat_path = os.path.relpath(os.path.join(loc, '../ground-truth', 'GT_' + fn.replace('jpg', 'mat')))

    points = loadmat(mat_path)['image_info'][0,0][0,0][0].astype(np.float32)

    # get the coordinates of head points within the image, because there's some

    # error point annotation which is out of the image region

    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)

    points = points[idx_mask]

    

    # zoom in or zoom out image to make sure its height and width are within min_size and max_size

    # rr: zooming ratio of the image

    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)

    im = np.array(im)

    if rr != 1.0:

        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)

        points = points * rr

    return Image.fromarray(im), points



def parse_args():

    parser = argparse.ArgumentParser(description='Test ')

    parser.add_argument('--origin-dir', default='../input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/images/',

                        help='original data directory')

    parser.add_argument('--data-dir', default='../input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/',

                        help='processed data directory')

    args = parser.parse_args()

    return args





class STRUCTS:

    def __init__(self, **kwargs):

        self.__dict__.update(**kwargs)

        

np.random.seed(11)



# Transfer 

if __name__ == '__main__':

    args = STRUCTS(

        origin_dir='../input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/images/',

        data_dir='../kaggle/working/part_B'

        

    )

    save_dir = args.data_dir

    # min_size max _size need to be set manually

    min_size = 512

    max_size = 1024

    # 1. Resize image to let make both height and width within min_size and max_size. And save the resized image

    # 2. Get the nearest distance of a point's neighbor point. And save the data.

    for phase in ['Train', 'Test']:

        sub_dir = os.path.join(args.origin_dir)

        if phase == 'Train':

            sub_phase_list = ['train', 'val']

            for sub_phase in sub_phase_list:

                sub_save_dir = os.path.join(save_dir, sub_phase)

                if not os.path.exists(sub_save_dir):

                    os.makedirs(sub_save_dir)

                

                np.random.seed(11)

                im_list = glob(os.path.join(sub_dir, '*jpg'))

                np.random.shuffle(im_list)

                

                if sub_phase == 'train':

                    for img_path in im_list[:320]:

                        im_path = img_path

                        name = os.path.basename(im_path)

                        print(name)

                        im, points = generate_data(im_path)

                        if sub_phase == 'train':

                            dis = find_dis(points, name)

                            points = np.concatenate((points, dis), axis=1)

                        im_save_path = os.path.join(sub_save_dir, name)

                        im.save(im_save_path)

                        print(im_save_path)

                        gd_save_path = im_save_path.replace('jpg', 'npy')

                        np.save(gd_save_path, points)

                

                else:

                    for img_path in im_list[320:]:

                        im_path = img_path

                        name = os.path.basename(im_path)

                        print(name)

                        im, points = generate_data(im_path)

                        if sub_phase == 'train':

                            dis = find_dis(points)

                            points = np.concatenate((points, dis), axis=1)

                        im_save_path = os.path.join(sub_save_dir, name)

                        im.save(im_save_path)

                        gd_save_path = im_save_path.replace('jpg', 'npy')

                        np.save(gd_save_path, points)

                        

        else:

            sub_save_dir = os.path.join(save_dir, 'test')

            if not os.path.exists(sub_save_dir):

                os.makedirs(sub_save_dir)

            im_list = glob(os.path.join(sub_dir, '../../test_data','*jpg'))

            for im_path in im_list:

                name = os.path.basename(im_path)

                print(name)

                im, points = generate_data(im_path)

                im_save_path = os.path.join(sub_save_dir, name)

                im.save(im_save_path)

                gd_save_path = im_save_path.replace('jpg', 'npy')

                np.save(gd_save_path, points)

# ../input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/ground-truth/GT_IMG_244.mat
# Loss.py

import tensorflow as tf



def split_tensor(tensor, sizes):

    sizes = tf.reshape(sizes, [-1])

    sizes = tf.cast(sizes, dtype=tf.int32)

    out_shape = [None] + tensor.shape[1:]

    out = tf.TensorArray(dtype=tensor.dtype, size=0, dynamic_size=True, infer_shape=False, element_shape=out_shape)

    out = out.split(tensor, sizes)

    return out







class BayLoss:

    def __init__(self, sigma, c_size, stride, background_ratio, use_background):

        assert c_size % stride == 0

        self.sigma = sigma

        self.bg_ratio = background_ratio

        # coordinate is same to image space, set to constant since crop size is same

        self.cood = tf.range(c_size, delta=stride, dtype=tf.float32) + stride / 2.0

        self.cood = tf.expand_dims(self.cood, axis=0)

        self.cood2 = self.cood * self.cood

        self.use_bg = use_background



    def _get_info(self, y_true):

        num_points_per_image = y_true[:, -1, 0]

        st_sizes = y_true[:, -1, 1]

        batch_size = tf.shape(y_true)[0]

        all_points = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)



        def info_cond(idx, all_points):

            return idx < batch_size



        def info_body(idx, all_points):

            length = tf.reshape(tf.cast(num_points_per_image[idx], dtype=tf.int32), (1,))[0]

            slices = tf.stack([1, length, 2])

            starts = tf.stack([idx, 0, 0])

            points = tf.slice(y_true, starts, slices)[0]

            all_points = all_points.write(idx, points)

            return idx + 1, all_points

        _, all_points = tf.while_loop(info_cond, info_body, loop_vars=[0, all_points])

        return num_points_per_image, all_points.concat(), st_sizes



    def _process_y_true(self, y_true):

        old_shape = tf.shape(y_true)[:2]

        ext_shape = tf.ones(2 - tf.shape(old_shape), dtype=tf.int32)

        new_shape = tf.concat([old_shape, ext_shape, [-1]], axis=0)

        y_true = tf.reshape(y_true, new_shape)

        zero_padding = 2 - tf.shape(y_true)[2]

        y_true = tf.pad(y_true, [(0, 0), (0, 0), (zero_padding, 0)])

        return y_true



    def __call__(self, y_true, y_pred):

        y_true = self._process_y_true(y_true)

        num_points_per_image, all_points, st_sizes = self._get_info(y_true)



        def bt0():

            x = all_points[:, 0, None]

            y = all_points[:, 1, None]

            x_dis = -2 * tf.matmul(x, self.cood) + x * x + self.cood2

            y_dis = -2 * tf.matmul(y, self.cood) + y * y + self.cood2

            y_dis = tf.expand_dims(y_dis, axis=2)

            x_dis = tf.expand_dims(x_dis, axis=1)

            dis = y_dis + x_dis

            dis_len = tf.shape(dis)[0]

            dis = tf.reshape(dis, (dis_len, -1))

            dis_list = split_tensor(dis, num_points_per_image)

            dis_list_size = dis_list.size()



            def cond(idx, loss):

                return idx < dis_list_size



            def body(idx, loss):

                dis = dis_list.read(idx)



                def f1():

                    N = tf.shape(dis)[0]

                    target = tf.ones(N, dtype=tf.float32)

                    if self.use_bg:

                        min_dis = tf.maximum(tf.reduce_min(dis, axis=0, keepdims=True)[0], 0)

                        d = st_sizes[idx] * self.bg_ratio

                        bg_dis = tf.square(d - tf.sqrt(min_dis))

                        bg_dis = tf.expand_dims(bg_dis, axis=0)

                        new_dis = tf.concat([dis, bg_dis], axis=0)

                        target = tf.pad(target, [(0, 1)])

                    else:

                        new_dis = dis

                    new_dis = -new_dis / (2.0 * self.sigma ** 2)

                    prob = tf.nn.softmax(new_dis, axis=0)

                    pre_count = tf.reduce_sum(tf.reshape(y_pred[idx], (1, -1)) * prob, axis=1)

                    return tf.reduce_sum(tf.abs(pre_count - target))



                def f2():

                    return tf.reduce_sum(tf.abs(y_pred[idx]))

                loss = loss + tf.cond(tf.greater(tf.shape(dis)[0], 0), f1, f2)

                return idx + 1, loss

            _, loss = tf.while_loop(cond, body, loop_vars=[0, 0.0])

            loss = loss / tf.cast(dis_list_size, dtype=tf.float32)

            return loss



        def eq0():

            return tf.reduce_mean(tf.reduce_sum(y_pred, axis=(1, 2, 3)))



        total_loss = tf.cond(tf.greater(tf.shape(all_points)[0], 0), bt0, eq0)

        return total_loss



class BayLoss2:

    def __init__(self, use_background, input_shape, downsampling=8):

        self.use_bg = use_background

        self.input_shape = input_shape

        self.downsampling = downsampling

    

    def _get_prob(self, y_true):

        num_points_per_image = y_true[:, -1, 0]

        batch_size = tf.shape(y_true)[0]

        prob_list = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, infer_shape=False)



        def prob_cond(idx, prob_list):

            return idx < batch_size



        def prob_body(idx, prob_list):

            length = tf.reshape(tf.cast(num_points_per_image[idx], dtype=tf.int32), (1,))[0]

            slices = tf.stack([1, length, -1])

            starts = tf.stack([idx, 0, 0])

            prob = tf.slice(y_true, starts, slices)[0]

            prob_list = prob_list.write(idx, prob)

            return idx + 1, prob_list

        _, prob_list = tf.while_loop(prob_cond, prob_body, loop_vars=[0, prob_list])

        return prob_list



    def _process_y_true(self, y_true):

        old_shape = tf.shape(y_true)[:2]

        out_ss = tf.shape(old_shape)

        ext_shape = tf.ones(2 - out_ss, dtype=tf.int32)

        new_shape = tf.concat([old_shape, ext_shape, [-1]], axis=0)

        y_true = tf.reshape(y_true, new_shape)

        zero_padding = tf.cast((self.input_shape//self.downsampling)**2 - tf.shape(y_true)[2], dtype='int32') ##?????????????????

        # ??????

        y_true = tf.pad(y_true, [(0, 0), (0, 0), (zero_padding, 0)])

        return y_true

    # input shape of y_true: (batch_size, max_num_points, flatten probability size)

    def __call__(self, y_true, y_pred):

        y_true = self._process_y_true(y_true)

        prob_list = self._get_prob(y_true)

        batch_size = tf.shape(y_true)[0]

        def cond(idx, loss):

            return idx < batch_size

        def body(idx, loss):

            prob = prob_list.read(idx)

            size = tf.shape(prob)[0]

            if self.use_bg:

                target = tf.ones(tf.maximum(size - 1, 0), dtype=tf.float32)

                target = tf.pad(target, [(0, 1)])

            else:

                target = tf.ones(size, dtype=tf.float32)

            padding = tf.cast(tf.equal(size, 0), dtype=tf.float32)

            prob = tf.pad(prob, [(0, padding), (0, 0)], constant_values=1)

            pre_count = tf.reduce_sum(tf.reshape(y_pred[idx], (1, -1)) * prob, axis=1)

            loss = loss + tf.reduce_sum(tf.abs(pre_count - target))

            return idx + 1, loss

                

        _, loss = tf.while_loop(cond, body, loop_vars=[0, 0.0])

        loss = loss / tf.cast(batch_size, dtype=tf.float32)

        return loss
# DataManager.py

import glob

import os

import cv2

import numpy as np

from tensorflow.keras.utils import Sequence

# from utils_func import imread



# read image and normalize to VGG16 training data mean.

def imread(im_path):

    img = cv2.imread(im_path)

    img = img / 255.

    img = (img - [0.406, 0.456, 0.485]) / [0.225, 0.224, 0.229]

    img = img.astype(np.float32)

    return img



def softmax(x, axis=-1):

    ndim = np.ndim(x)

    if ndim >= 2:

        y = np.exp(x - np.max(x, axis=axis, keepdims=True))

        return y / np.sum(y, axis, keepdims=True)

    else:

        raise ValueError('Cannot apply softmax to a tensor that is 1D. '

                         'Received input: %s' % x)



def random_crop(im_h, im_w, crop_h, crop_w):

    res_h = im_h - crop_h

    res_w = im_w - crop_w

    i = np.random.randint(0, res_h)

    j = np.random.randint(0, res_w)

    return i, j, crop_h, crop_w



def cal_innner_area(c_left, c_up, c_right, c_down, bbox):

    inner_left = np.maximum(c_left, bbox[:, 0])

    inner_up = np.maximum(c_up, bbox[:, 1])

    inner_right = np.minimum(c_right, bbox[:, 2])

    inner_down = np.minimum(c_down, bbox[:, 3])

    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)

    return inner_area



class BL_CrowdSequence(Sequence):

    def __init__(self, method, batch_size, sigma, c_size, stride, background_ratio, use_background):

        if method == 'train':

            self.root_path = '../kaggle/working/part_B/train'

            self.batch_size = batch_size

        elif method == 'valid':

            self.root_path = '../kaggle/working/part_B/val'

            self.batch_size = 1

        self.method = method

        self.c_size = c_size

        self.d_size = stride

        self.sigma = sigma

        self.bg_ratio = background_ratio

        self.use_bg = use_background

        self.cood = (np.arange(0, self.c_size, self.d_size, dtype=np.float32) + self.d_size / 2.0)[None]

        assert self.c_size % self.d_size == 0

        self.dc_size = self.c_size // self.d_size

        self.im_list = sorted(glob.glob(os.path.join(self.root_path, '*.jpg')))

    

    def train_transform(self, img, keypoints):

        """random crop image patch and find people in it"""

        ht, wd = img.shape[:2]

        st_size = min(wd, ht)

        assert st_size >= self.c_size

        assert len(keypoints) > 0

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)

        img = img[i:i+h, j:j+w]



        nearest_dis = np.clip(0.8 * keypoints[:, 2], 4.0, 40.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0

        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0

        bbox = np.concatenate((points_left_up, points_right_down), axis=1)

        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)

        origin_area = nearest_dis * nearest_dis

        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)

        mask = (ratio >= 0.5)

        keypoints = keypoints[mask]

        keypoints = keypoints[:, :2] - [j, i]  # change coodinate



        if len(keypoints) > 0:

            if np.random.random() > 0.5:

                img = img[:, ::-1]

                keypoints[:, 0] = w - keypoints[:, 0]

        else:

            if np.random.random() > 0.5:

                img = img[:, ::-1, :]

        return img, keypoints.astype(np.float32), st_size



    def __len__(self):

        return len(self.im_list) // self.batch_size



    def _expand_data(self, y_true, max_len):

        Y = []

        for points, st_size in y_true:

            y = np.pad(points, [(0, max_len + 1 - len(points)), (0, 0)])

            y[-1] = (len(points), st_size)

            Y.append(y)

        return Y

    

    def __getitem__(self, idx):

        if self.method == "train" and idx == 0:

            np.random.shuffle(self.im_list)

        X, Y = [], []

        max_len = 0

        for i in range(self.batch_size):

            im_path = self.im_list[(idx * self.batch_size + i) % len(self.im_list)]

            gd_path = im_path.replace('jpg', 'npy')

            img = imread(im_path)

            print(im_path)

            keypoints = np.load(gd_path)

            if self.method == 'train':

                x, y, z = self.train_transform(img, keypoints.copy())

                X.append(x)

                max_len = max(max_len, len(y))

                Y.append((y, z))

            elif self.method == 'valid':

                X.append(img)

                Y.append(len(keypoints))

        

        if self.method == 'train':

            Y = self._expand_data(Y, max_len)

        X = np.stack(X)

        Y = np.stack(Y)

        return X, Y



class BL_CrowdGenerator:

    def __init__(self, sigma, crop_size, stride, background_ratio, use_background):

        self.sigma = sigma

        self.c_size = crop_size

        self.divide = stride

        self.bg_ratio = background_ratio

        self.use_bg = use_background

    def GetFlow(self, batch_size, is_valid=False):

        method = 'valid' if is_valid else 'train'

        return BL_CrowdSequence(method, batch_size, self.sigma, self.c_size, self.divide, self.bg_ratio, self.use_bg)



class BL_CrowdSequence2(Sequence):

    def __init__(self, method, batch_size, sigma, c_size, stride, background_ratio, use_background):

        if method == 'train':

            self.root_path = '../kaggle/working/part_B/train'

        elif method == 'valid':

            self.root_path = '../kaggle/working/part_B/val'

            

            

        self.batch_size = batch_size

        # string: 'train' or 'valid'

        self.method = method

        # cropped image size

        self.c_size = c_size

        # downsampling ratio

        self.d_size = stride

        self.sigma = sigma

        # distance of background

        self.bg_ratio = background_ratio

        self.use_bg = use_background

        # get image coordinate(can be used as x or y coordinate, because the image size is square.)

        self.cood = (np.arange(0, self.c_size, self.d_size, dtype=np.float32) + self.d_size / 2.0)[None]

        assert self.c_size % self.d_size == 0

        self.dc_size = self.c_size // self.d_size

        self.im_list = sorted(glob.glob(os.path.join(self.root_path, '*.jpg')))

        print('img path list len: ', len(self.im_list))

    def train_transform(self, img, keypoints):

        """random crop image patch and find people in it"""

        ht, wd = img.shape[:2]

        st_size = min(wd, ht)

        assert st_size >= self.c_size

        assert len(keypoints) > 0

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)

        img = img[i:i+h, j:j+w]

        

        # Generate a bounding box according to 80% of the nearest distance to its neighbor point

        nearest_dis = np.clip(0.8 * keypoints[:, 2], 4.0, 40.0)

        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0

        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0

        bbox = np.concatenate((points_left_up, points_right_down), axis=1)

        inner_area = cal_innner_area(j, i, j+w, i+h, bbox)

        origin_area = nearest_dis * nearest_dis

        

        # The bounding box of each point would be cropped by our selected area, if the bounding box

        # of the point is minifeid less then 50% of its origin size, the discard that point.

        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)

        mask = (ratio >= 0.5)

        keypoints = keypoints[mask]

        

        # Normalize the coordinate of the keypoints, make the upper left be origin.

        keypoints = keypoints[:, :2] - [j, i]  # change coodinate



        if len(keypoints) > 0:

            # Data Augmentation: random horizontal flip

            if np.random.random() > 0.5:

                img = img[:, ::-1]

                keypoints[:, 0] = w - keypoints[:, 0]

        else:

            # Data Augmentation: random horizontal flip

            if np.random.random() > 0.5:

                img = img[:, ::-1, :]

        return img, keypoints.astype(np.float32), st_size



    def __len__(self):

        return len(self.im_list) // self.batch_size



    def _expand_data(self, y_true, max_len):

        Y = []

        for points, st_size in y_true:

            y = np.zeros((max_len + 1, 2), dtype=np.float32)

            size = len(points)

            y[:size] = points

            y[-1] = (size, st_size)

            Y.append(y)

        return Y

    

    def _process_training(self, y_true):

        all_points, st_sizes, slices = [], [], [0]

        for i in y_true:

            points, st_size = i

            all_points.append(points)

            st_sizes.append(st_size)

            slices.append(slices[-1] + len(points))

        slices = slices[1:][:-1]

        all_points = np.concatenate(all_points, axis=0)

        st_sizes = np.stack(st_sizes)

        max_len = 0

        if len(all_points) > 0:

            # calculate every head point to the every subsampling pixel's distance.

            # For Example:

            # given a 80x80 one-channel-image, a point (9,9), and downsampling rate 8

            # the corresponding subsampling pixel would be [4, 12, 16, ..... 60, 68, 76]

            # so we calculate the point to each subsampling pixel and generate distance map which shape is (1, 10*10).

            # Note that the number of point N, would result in distance map in shape (N, 10*10)

            x = all_points[:, 0, None]

            y = all_points[:, 1, None]

            x_dis = -2 * np.matmul(x, self.cood) + x * x + self.cood * self.cood

            y_dis = -2 * np.matmul(y, self.cood) + y * y + self.cood * self.cood

            y_dis = y_dis[:, :, None]

            x_dis = x_dis[:, None, :]

            dis = x_dis + y_dis

            dis = np.reshape(dis, (len(dis), -1))

            

            # split the distance map according to batch size.

            dis_list = np.split(dis, slices)

            prob_list = []

            for st_size, dis in zip(st_sizes, dis_list):

                if len(dis) > 0:

                    if self.use_bg:

                        min_dis = np.maximum(np.min(dis, axis=0, keepdims=True)[0], 0)

                        d = st_size * self.bg_ratio

                        bg_dis = ((d - np.sqrt(min_dis))**2)[None]

                        dis = np.concatenate([dis, bg_dis], axis=0)

                        

                    # according to each pixel to every points' distance, generate the probability map

                    dis = -dis / (2.0 * self.sigma ** 2)

                    # Note that the sum of each probability map in the same location of pixel is 1.0 

                    # For Example:

                    # If we get 10 probability with size of 56x56, then sum(probability_map[:, y_location, x_location])=1.0 

                    prob = softmax(dis, axis=0).astype(np.float32)

                else:

                    prob = dis 

                max_len = max(max_len, len(prob))

                prob_list.append(prob)

        else:

            prob_list = []

            for i in range(len(y_true)):

                prob_list.append(np.zeros((0, self.cood.shape[1]**2), dtype=np.float32))

                

        # Becaus different image would have different number of people(number of people equal to number of probability map),

        # we would make each data have same size by padding empty probability map, and last first pixel of the last probality map

        # would mark the actual number of the non-empty probability map.

        for i in range(len(prob_list)):

            prob = prob_list[i]

            size = len(prob)

            prob = np.pad(prob, [(0, max_len + 1 - size), (0, 0)], mode='constant')

            prob[-1, 0] = size

            prob_list[i] = prob

        return prob_list

    

    def __getitem__(self, idx):

        X, Y, max_len = [], [], 0

        for i in range(self.batch_size):

            im_path = self.im_list[(idx * self.batch_size + i) % len(self.im_list)]

            gd_path = im_path.replace('jpg', 'npy')

            # read and normalize image to imagenet distribution

            img = imread(im_path)

            keypoints = np.load(gd_path)

            if self.method == 'train':

                x, y, z = self.train_transform(img, keypoints.copy())

                X.append(x)

                Y.append((y, z))

            elif self.method == 'valid':

                X.append(img)

                Y.append(len(keypoints))

        

        if self.method == 'train':

            Y = self._process_training(Y)

        X = np.stack(X)

        Y = np.stack(Y)

        return X, Y



class BL_CrowdGenerator2:

    def __init__(self, sigma, crop_size, stride, background_ratio, use_background):

        self.sigma = sigma

        self.c_size = crop_size

        self.divide = stride

        self.bg_ratio = background_ratio

        self.use_bg = use_background

    def GetFlow(self, batch_size, is_valid=False):

        method = 'valid' if is_valid else 'train'

        return BL_CrowdSequence2(method, batch_size, self.sigma, self.c_size, self.divide, self.bg_ratio, self.use_bg)
import os

import tensorboard

import tensorflow as tf

import numpy as np

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import *

from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras import Model

from tensorflow.keras.applications import VGG16, VGG19

import matplotlib.pyplot as plt

from matplotlib import cm as CM

from tensorflow.python.client import device_lib

print(tf.__version__)

print(device_lib.list_local_devices())

# policy = mixed_precision.Policy('mixed_float16')

# mixed_precision.set_policy(policy)



# print('Compute dtype: %s' % policy.compute_dtype)

# print('Variable dtype: %s\n\n' % policy.variable_dtype)



for info in device_lib.list_local_devices():

    if (info.name.find('GPU') != -1):

        print(info)



def plot_history(history, name):

    history = history.history

    legend = []

    if 'loss' in history:

        plt.plot(history['loss'])

        legend.append('Train')

    if 'val_loss' in history:

        plt.plot(history['val_loss'])

        legend.append('Valid')

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(legend, loc='upper left')

    plt.savefig(name)

    

def get_callbacks(save_path, monitor='val_loss'):

    os.makedirs(os.path.split(save_path)[0], exist_ok=True)

    

    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_density_mae', save_weights_only=True, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=10, verbose=1, rate=0.3162)

    early_stopping = EarlyStopping(monitor=monitor, patience=30, verbose=1)

    

    return [reduce_lr, checkpoint, early_stopping]



def validation_loss(y_true, y_pred):

    v1, v2 = y_true, tf.reduce_sum(y_pred, axis=(1,2))

    return 2.0 * K.square(v1 - v2) + K.abs(v1 - v2)



def model_loss(training_loss, validation_loss):

    def _loss(y_true, y_pred):

        return tf.cond(tf.equal(tf.rank(y_true), 1), 

                       lambda: validation_loss(y_true, y_pred),

                       lambda: training_loss(y_true, y_pred)

                      )

    return _loss



def model_loss2(training_loss, validation_loss):

    def _loss(y_true, y_pred):

        return tf.cond(tf.equal(tf.rank(y_true), 3), 

                       lambda: training_loss(y_true, y_pred),

                       lambda: validation_loss(y_true, y_pred)

                      )

    return _loss



def process_y_true(y_true):

    old_shape = tf.shape(y_true)[:2]

    ext_shape = tf.ones(2 - tf.shape(old_shape), dtype=tf.int32)

    new_shape = tf.concat([old_shape, ext_shape, [-1]], axis=0)

    y_true = tf.reshape(y_true, new_shape)

    zero_padding = tf.cast((cfg.crop_size//cfg.downsample_ratio)**2 - tf.shape(y_true)[2], dtype='int32')

    y_true = tf.pad(y_true, [(0, 0), (0, 0), (0, zero_padding)])

    return y_true



def MAE(y_true, y_pred):

    y_true = process_y_true(y_true)

    gd_count = y_true[:, -1, 0]

    pre_count = tf.reduce_sum(y_pred, axis=(1, 2))

    res = pre_count - gd_count

    return tf.reduce_mean(tf.abs(res))



def MSE(y_true, y_pred):

    y_true = process_y_true(y_true)

    gd_count = y_true[:, -1, 0]

    pre_count = tf.reduce_sum(y_pred, axis=(1, 2))

    res = pre_count - gd_count

    return tf.reduce_mean(tf.square(res))



def show_images(images, cols = 2, titles = None, padding=1, axis="off", channel1=CM.jet):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    if titles is None: titles = [None for i in range(1,n_images + 1)]

    fig = plt.figure()

    

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)



        plt.axis(axis)

        plt.subplots_adjust(wspace=padding, hspace=padding)



        if (image.shape[2] == 1):

            image = image[:,:,0]

            plt.imshow(image, cmap=channel1)

        elif np.any(image > 1.0):

            plt.imshow(image / 255.0)

        else:

            plt.imshow(image)

        a.set_title(title, fontsize=20)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.show()





def VGG19_counter(shape=(None, None, 3)):

    vgg19 = VGG19(include_top=False, input_shape=shape)

    input_layer=vgg19.layers[0].input

    x=vgg19.layers[20].output

    

    # Block 6

    x = UpSampling2D(size=(2, 2), name='block5_up')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv1')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block6_conv2')(x)

    x = Conv2D(  1, (3, 3), activation='relu', padding='same', name='block6_conv3')(x)

    

    model = Model(input_layer, x, name='BL_model')

    return model
class STRUCT:

    def __init__(self, **kwargs):

        self.__dict__.update(**kwargs)



cfg = STRUCT(

    sigma=8,

    crop_size=512,

    downsample_ratio=8,

    background_ratio=0.15,

    use_background=True,

    batch_size=4

)



gen = BL_CrowdGenerator2(sigma=cfg.sigma,

                        crop_size=cfg.crop_size,

                        stride=cfg.downsample_ratio,

                        background_ratio=cfg.background_ratio,

                        use_background=cfg.use_background)



train = gen.GetFlow(batch_size=cfg.batch_size, is_valid=False)

valid = gen.GetFlow(batch_size=1, is_valid=True)

print('train gen: ', len(train))

print('valid gen: ', len(valid))



np.random.seed(11)

a,b = train[2]

b = np.reshape(b, (b.shape[0],b.shape[1],int(np.sqrt(b.shape[2])),int(np.sqrt(b.shape[2])),1))

for i in range(2):

    show_images([a[i]],1)

    show_images(b[i,:16],4, ["max: {}, min: {}, sum: {}".format(np.max(i), np.min(i), np.sum(i)) for i in b[i,:16]])
from tensorflow.keras.initializers import *

def VGG19_counter(shape=(None, None, 3)):

    init=RandomNormal(stddev=0.01)

    vgg19 = VGG19(include_top=False, input_shape=shape)

    vgg19.trainable=False

    input_layer=vgg19.layers[0].input

    x=vgg19.layers[20].output

    

    # Block 6

    x = UpSampling2D(size=(2, 2), name='block5_up')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block6_conv1', kernel_initializer=init)(x)

    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block6_conv2', kernel_initializer=init)(x)

    x = BatchNormalization()(x)

    x = Conv2D(  1, (3, 3), activation='sigmoid', padding='same', name='block6_conv3', kernel_initializer=init)(x)

    

    model = Model(input_layer, x, name='BL_model')

    return model



training_loss = BayLoss2(use_background=cfg.use_background, input_shape=cfg.crop_size, downsampling=cfg.downsample_ratio)

loss = model_loss2(training_loss, validation_loss)



model = VGG19_counter((cfg.crop_size, cfg.crop_size, 3))

model.compile(loss=loss, optimizer=Adam(1e-3, decay=0.0001), metrics=[MAE])

model.summary()
print(len(train))

monitor='val_loss'

save_path='./vgg19_bl.h5'

os.makedirs(os.path.split(save_path)[0], exist_ok=True)



checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_density_mae', save_weights_only=True, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=20, verbose=1, rate=0.3162)

early_stopping = EarlyStopping(monitor=monitor, patience=80, verbose=1)



his1 = model.fit(train, validation_data=valid, 

                    epochs=400, verbose=1, workers=4,

                    callbacks=[checkpoint, reduce_lr, early_stopping])



plot_history(his1, './vgg19_bl_loss_w_val.png')

model.save_weights('./final_vgg19_bl.h5')
for i in range(20): 

    a,b = valid[i]

    img = (a * [0.225, 0.224, 0.229]) + [0.406, 0.456, 0.485]

    img = np.clip(img, a_max=1.0, a_min=0.0)

    c = model.predict(a)

    

    show_images([img[0], c[0]], 1, [b[0], np.sum(c)])

    