import os, sys
import cv2
import random
import numpy as np
from detectBlobs import DetectBlobs

# detectKeypoints(...): Detect feature keypoints in the input image
#   You can either reuse your blob detector from part 1 of this assignment
#   or you can use the provided compiled blob detector detectBlobsSolution.pyc
#
#   Input: 
#        im  - input image
#   Output: 
#        detected feature points (in any format you like).

def detectKeypoints(im):
    # YOUR CODE STARTS HERE
    max = 1500
    blobs = []
    for blob in DetectBlobs(im):
        y = blob[0]
        x = blob[1]
        radius = blob[2]
        score = blob[3]
        blobs.append({"x":x, "y":y, "radius":radius, "score":score})
    blobs = sorted(blobs, key=lambda x: -x["score"])[:max]
    return blobs


# computeDescriptors(...): compute descriptors from the detected keypoints
#   You can build the descriptors by flatting the pixels in the local 
#   neighborhood of each keypoint, or by using the SIFT feature descriptors from
#   OpenCV (see computeSIFTDescriptors(...)). Use the detected blob radii to
#   define the neighborhood scales.
#
#   Input:
#        im          - input image
#        keypoints   - detected feature points
#
#   Output:
#        descriptors - n x dim array, where n is the number of keypoints 
#                      and dim is the dimension of each descriptor. 
#
def computeDescriptors(im, keypoints):
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    blobs = []
    for kpt in keypoints:
        blobs.append([kpt['y'], kpt['x'], kpt['radius'], kpt['score']])
    
    blobs = np.stack(blobs)
    return computeSIFTDescriptors(im, blobs)


# computeSIFTDescriptors(...): compute SIFT feature descriptors from the
#   detected keypoints. This function is provided to you.
#
#   Input:
#        im          - H x W array, the input image
#        keypoints   - n x 4 array, where there are n blobs detected and
#                      each row is [x, y, radius, score]
#
#   Output:
#        descriptors - n x 128 array, where n is the number of keypoints
#                      and 128 is the dimension of each descriptor.
#
def computeSIFTDescriptors(im, keypoints):
    kp = []
    for blob in keypoints:
        kp.append(cv2.KeyPoint(blob[1], blob[0], _size=blob[2]*2, _response=blob[-1], _class_id=len(kp)))
    detector = cv2.xfeatures2d_SIFT.create()
    return detector.compute(im, kp)[1]



# getMatches(...): match two groups of descriptors.
#
#   There are several strategies you can use to match keypoints from the left
#   image to those in the right image. Feel free to use any (or combinations
#   of) strategies:
#
#   - Return all putative matches. You can select all pairs whose
#   descriptor distances are below a specified threshold,
#   or select the top few hundred descriptor pairs with the
#   smallest pairwise distances.
#
#   - KNN-Match. For each keypoint in the left image, you can simply return the
#   the K best pairings with keypoints in the right image.
#
#   - Lowe's Ratio Test. For each pair of keypoints to be returned, if the
#   next best match with the same left keypoint is nearly as good as the
#   current match to be returned, then this match should be thrown out.
#   For example, given point A in the left image and its best and second best
#   matches B and C in the right image, we check: score(A,B) < score(A,C)*0.75
#   If this test fails, don't return pair (A,B)
#
#
#   Input:
#         descriptors1 - the descriptors of the first image
#         descriptors2 - the descriptors of the second image
# 
#   Output: 
#         index1       - 1-D array contains the indices of descriptors1 in matches
#         index2       - 1-D array contains the indices of descriptors2 in matches

def getMatches(descriptors1, descriptors2):
    distances_matrix = np.empty((descriptors1.shape[0], descriptors2.shape[0]))

    for i in range(distances_matrix.shape[0]):
        for j in range(distances_matrix.shape[1]):
            distances_matrix[i][j] = np.sqrt(np.sum((descriptors1[i] - descriptors2[j])**2))
    sort_indices = np.argsort(distances_matrix)

    index1 = []
    index2 = []

    for i in range(descriptors1.shape[0]):
        first = sort_indices[i][0]
        second = sort_indices[i][1]

        if distances_matrix[i][first] < distances_matrix[i][second] * 0.75:
            index1.append(i)
            index2.append(first)
    return (index1, index2)



# RANSAC(...): run the RANSAC algorithm to estimate a homography mapping between two images.
#   Input:
#        matches - two 1-D arrays that contain the indices on matches. 
#        keypoints1       - keypoints on the left image
#        keypoints2       - keypoints on the right image
#
#   Output:
#        H                - 3 x 3 array, a homography mapping between two images
#        numInliers       - int, the number of inliers 
#
#   Note: Use four matches to initialize the homography in each iteration.
#         You should output a single transformation that gets the most inliers 
#         in the course of all the iterations. For the various RANSAC parameters 
#         (number of iterations, inlier threshold), play around with a few 
#         "reasonable" values and pick the ones that work best.

def RANSAC(matches, keypoints1, keypoints2):
    N = 30
    s = 4
    thresh = 4

    idx1, idx2 = matches

    winning_H = []
    winning_inliers = 0

    # Begin RANSAC
    for n in range(N):
        randoms = np.random.rand(s) * len(idx1)
        randoms = randoms.astype(int)
        samples = []

        # Get s matches of the form (x_i, x_i') where x_i has an x and y coordinate
        for random in randoms:
            orig_kpt = (keypoints1[idx1[random]]['x'], keypoints1[idx1[random]]['y'])
            dst_kpt = (keypoints2[idx2[random]]['x'], keypoints2[idx2[random]]['y'])
            samples.append((orig_kpt, dst_kpt))

        H = get_H_matrix(samples)

        inliers = 0
        for i in range(len(idx1)):
            orig_kpt = np.array([keypoints1[idx1[i]]['x'], keypoints1[idx1[i]]['y'], 1])
            dst_kpt = np.array([keypoints2[idx2[i]]['x'], keypoints2[idx2[i]]['y'], 1])
            transf_kpt =  np.matmul(H, orig_kpt)
            transf_kpt = transf_kpt / transf_kpt[2]
            dist = np.sqrt(np.sum((transf_kpt - dst_kpt) ** 2))

            if dist < thresh:
                inliers += 1
        
        if inliers > winning_inliers:
            winning_inliers = inliers
            winning_H = H
        
    
    return winning_H, winning_inliers

# Helper function to compute the homography matrix H
def get_H_matrix(samples):
    # Create matrix A of size 2n x 9 where n is the number of samples
    A = []
    for i in range(len(samples)):
        orig_kpt = samples[i][0]
        dst_kpt = samples[i][1]

        x = [orig_kpt[0], orig_kpt[1], 1]
        x_prime = dst_kpt[0]
        y_prime = dst_kpt[1]
        
        line1 = [0, 0, 0, x[0], x[1], x[2], -y_prime * x[0], -y_prime * x[1], -y_prime]
        line2 = [x[0], x[1], x[2], 0, 0, 0, -x_prime * x[0], -x_prime * x[1], -x_prime]
        A.append(line1)
        A.append(line2)
    
    A = np.stack(A)
    
    # Compute eigenvectors and eigenvalues of AT*A and find smallest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(A.T, A))
    smallest_ev = eigenvectors[:, 0]
    return smallest_ev.reshape((3,3))



# warpImageWithMapping(...): warp one image using the homography mapping and
#   composite the warped image and another image into a panorama.
# 
#   Input: 
#        im_left, im_right - input images.
#        H                 - 3 x 3 array, a homography mapping
#  
#   Output:
#        Panorama made of the warped image and the other.
#
#       To display the full warped image, you may want to modify the matrix H.
#       CLUE: first get the coordinates of the corners under the transformation,
#             use the new corners to determine the offsets to move the
#             warped image such that it can be displayed completely.
#             Modify H to fulfill this translate operation.
#       You can use cv2.warpPerspective(...) to warp your image using H

def warpImageWithMapping(im_left, im_right, H):
    H_list= [H, np.eye(3)]
    images = [im_left, im_right]
    return warpImagesWithMapping(images, H_list)

# warpImageWithMapping(...): warp many images using the homography mapping and
#   composite the warped image and another image into a panorama.
# 
#   Input: 
#        im_list  list of images
#        H_list      list of 3 x 3 array, a homography mapping
#  
#   Output:
#        Panorama made of the warped images.

def warpImagesWithMapping(im_list, H_list):
    x_max = 0
    y_max = 0
    x_min = 0
    y_min = 0
    for im, H in zip(im_list, H_list):
        corners = np.array([
            [0, 0, 1],
            [0, im.shape[0], 1],
            [im.shape[1], 0, 1],
            [im.shape[1], im.shape[0], 1] 
        ]).T
        corners = np.matmul(H, corners)
        for i in range(4):
            x_max = max(x_max , corners[0][i] / corners[2][i])
            y_max = max(y_max , corners[1][i] / corners[2][i])
            x_min = min(x_min , corners[0][i] / corners[2][i])
            y_min = min(y_min , corners[1][i] / corners[2][i])
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    output_imgs = []
    for im, H in zip(im_list, H_list):
        output_imgs.append(cv2.warpPerspective(im, np.matmul(translation, H), ((int)(x_max - x_min), (int)(y_max - y_min))))
    return (np.sum(output_imgs, axis=0) / 2).astype(int)



# drawMatches(...): draw matches between the two images and display the image.
#
#   Input:
#         im1: input image on the left
#         im2: input image on the right
#         matches: (1-D array, 1-D array) that contains indices of descriptors in matches
#         keypoints1: keypoints on the left image
#         keypoints2: keypoints on the right image
#         title: title of the displayed image.
#
#   Note: This is a utility function that is provided to you. Feel free to
#   modify the code to adapt to the keypoints and matches in your own format.

def drawMatches(im1, im2, matches, keypoints1, keypoints2, title='matches'):
    idx1, idx2 = matches
    
    cv2matches = []
    for i,j in zip(idx1, idx2):
        cv2matches.append(cv2.DMatch(i, j, _distance=0))

    _kp1, _kp2 = [], []
    for i in range(len(keypoints1)):
        _kp1.append(cv2.KeyPoint(keypoints1[i]['x'], keypoints1[i]['y'], _size=keypoints1[i]['radius'], _response=keypoints1[i]['score'], _class_id=len(_kp1)))
    for i in range(len(keypoints2)):
        _kp2.append(cv2.KeyPoint(keypoints2[i]['x'], keypoints2[i]['y'], _size=keypoints2[i]['radius'], _response=keypoints2[i]['score'], _class_id=len(_kp2)))
    
    im_matches = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(im1, _kp1, im2, _kp2, cv2matches, im_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow(title, im_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
