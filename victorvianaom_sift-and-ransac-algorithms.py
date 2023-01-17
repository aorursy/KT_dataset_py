# import cv2

# from pylab import *

# from skimage import io

# import numpy as np

# import matplotlib.pyplot as plt



# def extract_sift_features(img):

#     sift_initialize = cv2.xfeatures2d.SIFT_create()

#     key_points, descriptors = sift_initialize.detectAndCompute(img, None)

#     return key_points, descriptors



# def showing_sift_features(img1, img2, key_points):

#     return plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))



# # THE BELLOW CODE DOESN'T WORK IF THE IMAGES HAVE UNICODE CHARACTERES IN THEIR PATH

# # OpenCV does not accept it

# # So the files are directly inside the '../input' folder I don't know why:

# img1 = cv2.imread('../input/tajone.jpg')

# img2 = cv2.imread('../input/tajtwo.jpg')



# # converting to gray:

# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



# #Key points:

# img1_key_points, img1_descriptors = extract_sift_features(img1_gray)

# img2_key_points, img2_descriptors = extract_sift_features(img2_gray)



# print('Displaying SIFT features:')

# showing_sift_features(img1_gray, img1, img1_key_points)



# norm = cv2.NORM_L2

# bruteForce = cv2.BFMatcher(norm)

# matches = bruteForce.match(Image1_descriptors, Image2_descriptors)



# matches = sorted(matches, key = lambda match:match.distance)

# matched_img = cv2.drawMatches(

# Image1, Image1_key_points,

# Image2, Image2_key_points,

# matches[:100], Image2.copy())

# plt.figure(figsize=(100,300))

# plt.imshow(matched_img)
# ############## Ransac.py: contains the entire RANSAC algorithm...

# import numpy as np

# from Affine import *



# K=3

# threshold=1



# ITER_NUM = 2000



# def residual_lengths(X, Y, s, t):

#     e = np.dot(X, s) + Y

#     diff_square = np.power(e - t, 2)

#     residual = np.sqrt(np.sum(diff_square, axis=0))

#     return residual

# def ransac_fit(pts_s, pts_t):

#     inliers_num = 0

#     A = None

#     t = None

#     inliers = None

#     for i in range(ITER_NUM):

#         idx = np.random.randint(0, pts_s.shape[1], (K, 1))

#         A_tmp, t_tmp = estimate_affine(pts_s[:, idx], pts_t[:, idx])

#         residual = residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

#         if not(residual is None):

#             inliers_tmp = np.where(residual < threshold)

#             inliers_num_tmp = len(inliers_tmp[0])

#             if inliers_num_tmp > inliers_num:

#                 inliers_num = inliers_num_tmp

#                 inliers = inliers_tmp

#                 A = A_tmp

#                 t = t_tmp

#         else:

#             pass

#     return A, t, inliers



# ############## Affine.py:

# import numpy as np



# def estimate_affine(s, t):

#     num = s.shape[1]

#     M = np.zeros((2 * num, 6))

#     for i in range(num):

#         temp = [[s[0, i], s[1, i], 0, 0, 1, 0], [0, 0, s[0, i], s[1, i], 0, 1]]

#         M[2 * i: 2 * i + 2, :] = np.array(temp)

#     b = t.T.reshape((2 * num, 1))

#     theta = np.linalg.lstsq(M, b)[0]

#     X = theta[:4].reshape((2, 2))

#     Y = theta[4:]

#     return X, Y



# ############## Align.py:

# import numpy as np

# from Ransac import *

# import cv2

# from Affine import *



# def extract_SIFT(img):

#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     sift = cv2.xfeatures2d.SIFT_create()

#     kp, desc = sift.detectAndCompute(img_gray, None)

#     kp = np.array([p.pt for p in kp]).T

#     return kp, desc

# def match_SIFT(descriptor_source, descriptor_target):

#     bf = cv2.BFMatcher()

#     matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)

#     pos = np.array([], dtype=np.int32).reshape((0, 2))

#     matches_num = len(matches)

#     for i in range(matches_num):

#         if matches[i][0].distance <= 0.8 * matches[i][1].distance:

#             temp = np.array([matches[i][0].queryIdx, matches[i][0].trainIdx])

#             pos = np.vstack((pos, temp))

#     return pos

# def affine_matrix(s, t, pos):

#     s = s[:, pos[:, 0]]

#     t = t[:, pos[:, 1]]

#     _, _, inliers = ransac_fit(s, t)

#     s = s[:, inliers[0]]

#     t = t[:, inliers[0]]

#     A, t = estimate_affine(s, t)

#     M = np.hstack((A, t))

#     return M



# ############## Main Code:

# import numpy as np

# import cv2

# from Ransac import *

# from Affine import *

# from Align import *



# img_source = cv2.imread("2.jpg")

# img_target = cv2.imread("target.jpg")



# keypoint_source, descriptor_source = extract_SIFT(img_source)

# keypoint_target, descriptor_target = extract_SIFT(img_target)



# pos = match_SIFT(descriptor_source, descriptor_target)



# H = affine_matrix(keypoint_source, keypoint_target, pos)

# rows, cols, _ = img_target.shape

# warp = cv2.warpAffine(img_source, H, (cols, rows))

# merge = np.uint8(img_target * 0.5 + warp * 0.5)



# cv2.imshow('img', merge)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
print('tears in heaven')
import cv2

cv2.BFMatcher().knnMatch()