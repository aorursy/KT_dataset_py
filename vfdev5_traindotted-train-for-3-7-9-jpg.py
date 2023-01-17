import cv2

import matplotlib.pylab as plt



def plt_st(l1,l2):

    plt.figure(figsize=(l1,l2))
for image_id in ['3', '7', '9']:

    img1 = cv2.imread('../input/Train/' + image_id + '.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    img2 = cv2.imread('../input/TrainDotted/' + image_id + '.jpg')

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    

    plt_st(12, 10)

    plt.subplot(121)

    plt.imshow(img1)

    plt.subplot(122)

    plt.imshow(img2)

    plt.suptitle("Train VS TrainDotted, image id = %s.jpg" % image_id)

plt_st(12, 5)