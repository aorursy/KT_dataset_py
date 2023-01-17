# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from PIL import Image
from PIL import ImageFilter
import multiprocessing
import random; random.seed(2016);
import cv2
import re
import os, glob
files = pd.DataFrame([[f,int(f.split("/")[5][3:-5])] for f in glob.glob("../input/images/IMAGES/IMAGE SET 1/img*.jpeg")])
files.columns = ['path', 'pic_no']
files = files.sort_values(by=["pic_no"], ascending=[1]).reset_index(drop=True)
plt.rcParams['figure.figsize'] = (24.0, 24.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 1
a = []
img = cv2.imread(files.path[71])
for l in range(1,55):
    im = cv2.imread(files.path[70-l])
    vis = np.concatenate((img, im), axis=1)
    img = vis
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
def equalize_histogram_color(img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	return img
import sys
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map
# Find SIFT and return Homography Matrix
def get_kaze_homography(img1, img2):

	# Initialize SIFT 
	kaze = cv2.KAZE_create()
    

	# Extract keypoints and descriptors
	k1, d1 = kaze.detectAndCompute(calc_energy(img1), None)
	k2, d2 = kaze.detectAndCompute(calc_energy(img2), None)

	# Bruteforce matcher on the descriptors
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(d1,d2, k=2)

	# Make sure that the matches are good
	verify_ratio = 0.8 # Source: stackoverflow
	verified_matches = []
	for m1,m2 in matches:
		# Add to array only if it's a good match
		if m1.distance < 0.8 * m2.distance:
			verified_matches.append(m1)

	# Mimnum number of matches
	min_matches = 8
	if len(verified_matches) > min_matches:
		
		# Array to store matching points
		img1_pts = []
		img2_pts = []

		# Add matching points to array
		for match in verified_matches:
			img1_pts.append(k1[match.queryIdx].pt)
			img2_pts.append(k2[match.trainIdx].pt)
		img1_pts = np.float32(img1_pts).reshape(-1,1,2)
		img2_pts = np.float32(img2_pts).reshape(-1,1,2)
		
		# Compute homography matrix
		M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
		return M
	else:
		print('Error: Not enough matches')
def get_stitched_image(img1, img2, M):

	# Get width and height of input images	
	w1,h1 = img1.shape[:2]
	w2,h2 = img2.shape[:2]

	# Get the canvas dimesions
	img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
	img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


	# Get relative perspective of second image
	img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

	# Resulting dimensions
	result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

	# Getting images together
	# Calculate dimensions of match points
	[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	
	# Create output array after affine transformation 
	transform_dist = [-x_min,-y_min]
	transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 

	# Warp images to get the resulting image
	result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
									(x_max-x_min, y_max-y_min))
	result_img[transform_dist[1]:w1+transform_dist[1], 
				transform_dist[0]:h1+transform_dist[0]] = img1

	# Return the result
	return result_img
plt.rcParams['figure.figsize'] = (24.0, 24.0)
plt.subplots_adjust(wspace=0, hspace=0)
img1 = cv2.imread(files.path[69])
img2 = cv2.imread(files.path[68])
img1 = equalize_histogram_color(img1)
img2 = equalize_histogram_color(img2)
M =  get_kaze_homography(img1, img2)
	# Stitch the images together using homography matrix
result_image = get_stitched_image(img2, img1, M)
for l in range(1,2):
    img2 = cv2.imread(files.path[68-l])
    img1 = equalize_histogram_color(result_image)
    img2 = equalize_histogram_color(img2)
    M =  get_kaze_homography(img1, img2)
	# Stitch the images together using homography matrix
    result_image = get_stitched_image(img2, img1, M)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)); plt.axis('off')
img1 = cv2.imread(files.path[61])
img2 = cv2.imread(files.path[60])
img1 = equalize_histogram_color(img1)
img2 = equalize_histogram_color(img2)
M =  get_kaze_homography(img1, img2)
	# Stitch the images together using homography matrix
result_image = get_stitched_image(img2, img1, M)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)); plt.axis('off')

im1 = cv2.imread(files.path[69])
im2 = cv2.imread(files.path[68])
im3 = cv2.subtract(im2,im1)
vis = np.concatenate((im2,im1,im3),axis=1)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.axis('off')

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb, time 
def find_homography(img1_color, img2_color, H4p1=None, H4p2_gt=None, visual=False, method='ORB', min_match_count = 25, return_h_inv=True):
    """By default, return h_inv for synthetic data"""
    print('===> Image size:', img1_color.shape) 
    if len(img1_color.shape) == 3:
      img1 = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY) 
      img2 = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)
      if visual:
          img1_color = cv2.cvtColor(img1_color, cv2.COLOR_RGB2BGR)
          img2_color = cv2.cvtColor(img2_color, cv2.COLOR_RGB2BGR)
    else:
      img1 = img1_color
      img2 = img2_color
    
    # Create feature detectors
    if method=='ORB':
        ft_detector = cv2.ORB_create()
        # Create brute-force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    elif method=='SIFT':
        ft_detector = cv2.xfeatures2d.SIFT_create()
        # Create brute-force matcher object
        bf = cv2.BFMatcher(crossCheck=False)
    else:
      return  np.zeros([1,4,2]), np.eye(3), 0 

    keyPoints1, descriptors1 = ft_detector.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = ft_detector.detectAndCompute(img2, None)

    try:
      descriptors1 = descriptors1.astype(np.uint8)
      descriptors2 = descriptors2.astype(np.uint8)
    except:
      print('===> Error on finding descriptors') 
      return np.zeros([1,4,2]), np.eye(3), 1 
    
    
    if visual:
     print('===> No of keypoints:', len(keyPoints1), len(keyPoints2)) 
     kp_img1 = img1.copy() 
     kp_img1 = cv2.drawKeypoints(img1, keyPoints1, kp_img1)
     kp_img2 = img2.copy()
     kp_img2 = cv2.drawKeypoints(img2, keyPoints2, kp_img2)
     kp_pair = np.concatenate((kp_img1, kp_img2), axis=1)
     plt.imshow(kp_pair, cmap='gray') 
     plt.axis('off')
     plt.show()
     plt.pause(0.05)
    # Match the descriptors
    try:
        matches = bf.match(descriptors1, descriptors2)
    except:
        print('===> Error on Matching. Return identity matrix') 
        return np.zeros([1,4,2]), np.eye(3), 1 
   
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Check number of matches 
    if len(matches) <= 0:
      print('===> Not enough matched features. Return identity matrix')
      return np.zeros([1,4,2]), np.eye(3), 1 

    num_choosing = min_match_count+ 1 
    if num_choosing > len(matches):
        num_choosing = len(matches)
    
    #TODO: use all features or not
    # goodMatches = matches[0:num_choosing]
    
    # Synthetic:
    goodMatches = matches
    # Real:
    if not return_h_inv:
      goodMatches = matches[0:25]
    
    # Draw matches 
    if visual:
      #newKP2 = [keyPoints2[m.trainIdx] for m in goodMatches]
      print('===> No of matches, goodMatches:', len(matches), len(goodMatches)) 
      match_img = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches, None, flags=2)
      plt.imshow(match_img)
      plt.axis('off') 
      plt.show()
      plt.pause(0.05)
    # Apply the homography transformation if we have enough good matches 
    # print('===> Applying RANSAC...')

    if len(goodMatches) >= min_match_count:
        # Get the good key points positions
        sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
        # Obtain homography
        try:   
          M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
          matchesMask = mask.ravel().tolist()
        except:
          return np.zeros([1,4,2]), np.eye(3), 1
        try:
          if return_h_inv:
            M_inv = np.linalg.inv(M)
          else:
            M_inv = M 
        except:
          print('Error in inverse H', mask)
          return np.zeros([1,4,2]), np.eye(3), 1 
        
        # Apply the perspective transformation to the source image corners
        h, w = img1.shape
        if len(H4p1) > 0:
            corners = np.float32(H4p1).reshape(1,4,2)
            transformedCorners = cv2.perspectiveTransform(corners,M_inv)
        else:
            corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(1,4,2)
            transformedCorners = cv2.perspectiveTransform(corners, M_inv)


    else:
        print("-----Found - %d/%d" % (len(goodMatches), min_match_count))
        matchesMask = None
        return np.zeros([1,4,2]), np.eye(3), 1 

    # Draw the matches
    if visual:
        drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
        img_warped = cv2.warpPerspective(img1_color, M, (img1_color.shape[1], img1_color.shape[0])) 
        
        
        # Draw a polygon on the second image joining the transformed corners  
        img2_color = cv2.polylines(img2_color, [np.int32(transformedCorners)], True, (0, 100, 255),3, cv2.LINE_AA)
        img1_color = cv2.polylines(img1_color, [np.int32(corners)], True, (125, 55, 125),3, cv2.LINE_AA)
   
        try: 
            img2_color = cv2.polylines(img2_color, [np.int32(H4p2_gt)], True, (0, 255, 0),3, cv2.LINE_AA)
        except:
            pass 
        result = cv2.drawMatches(img1_color, keyPoints1, img2_color, keyPoints2, goodMatches, None, **drawParameters)
        # resize image 
        #result = cv2.resize(result, (1248, 188))
        #img_warped = cv2.resize(img_warped, (624,188))   
        ## Display the results
        #cv2.imshow('Homography', result)
        #cv2.imshow('Warped 1', img_warped)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        plt.subplot(3,1,1)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) #cmap='gray')
        plt.title('Matching')
        plt.axis('off')
        plt.subplot(3,1,2)
        plt.imshow(cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))#,cmap='gray')
        plt.title('Warped Image 1')
        plt.axis('off')
        plt.subplot(3,1,3)
        plt.imshow( cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))#,cmap='gray')
        plt.title('Image 2')
        plt.axis('off') 
        plt.show()
        plt.pause(0.05) 
    return transformedCorners - corners, M, 0 