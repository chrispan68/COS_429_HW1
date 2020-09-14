import numpy as np
import math
import cv2
from scipy import ndimage

# This code is taken and converted to Python from:
#
#   CMPSCI 670: Computer Vision, Fall 2014
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
# Part1:
#
#   DetectBlobs(...) detects blobs in the image using the Laplacian
#   of Gaussian filter. Blobs of different size are detected by scaling sigma
#   as well as the size of the filter or the size of the image. Downsampling
#   the image will be faster than upsampling the filter, but the decision of
#   how to implement this function is up to you.
#
#   For each filter scale or image scale and sigma, you will need to keep track of
#   the location and matching score for every blob detection. To combine the 2D maps
#   of blob detections for each scale and for each sigma into a single 2D map of
#   blob detections with varying radii and matching scores, you will need to use
#   Non-Max Suppression (NMS).
#
#   Additional Notes:
#       - We greyscale the input image for simplicity
#       - For a simple implementation of Non-Max-Suppression, you can suppress
#           all but the most likely detection within a sliding window over the
#           2D maps of blob detections (ndimage.maximum_filter may help).
#           To combine blob detections into a single 2D output,
#           you can take the max along the sigma and scale axes. If there are
#           still too many blobs detected, you can do a final NMS. Remember to
#           keep track of the blob radii.
#       - A tip that may improve your LoG filter: Normalize your LoG filter
#           values so that your blobs detections aren't biased towards larger
#           filters sizes
#
#   You can qualitatively evaluate your code using the evalBlobs.py script.
#
# Input:
#   im             - input image
#   sigma          - base sigma of the LoG filter
#   num_intervals  - number of sigma values for each filter size
#   threshold      - threshold for blob detection
#
# Ouput:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def DetectBlobs(
    im,
    sigma = 2,
    num_intervals = 12,
    threshold = 1.1e-2
    ):

    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # YOUR CODE STARTS HERE
    im_height = im.shape[0]
    im_width = im.shape[1]

    # Creates all the kernels for any given octave. (Reuse these kernels for each octave)
    dog_kernels = []
    kernels = []
    sigmas = []
    k = 2**(1/num_intervals)
    for i in range(-1, num_intervals+1):
        sigmas.append(sigma * k**i)
        kernels.append(gaussian(sigma * k**i,  (int)(12*sigma + 1)))
        if(i >= 0):
            dog_kernels.append(kernels[-1] - kernels[-2])

    # 1) Uses the difference of Gaussian kernels (Approximate Laplacian of Gaussian) 
    # 2) Convolves the dog_kernels with the downsampled image 
    # 3) Checks if the given pixel is the unique max in its neighborhood. 
    # 4) Checks if the given pixel is greater than the activation threshold. 
    # 
    # Result: min_max contains a 3d numpy array of booleans. Radii is a 1d list of radii. Scores is a 3d numpy array of floats. 
    scores = []
    min_max = []
    radii = []
    pow = 0
    while min(im_height, im_width)>= (3*sigma + 1) * 2**pow:
        resized = cv2.resize(im, (int(im_width / 2**pow) , int(im_height / 2**pow)), interpolation = cv2.INTER_AREA)
        DoG = []
        for i,kernel in enumerate(dog_kernels):
            conv = ndimage.convolve(resized, kernel)
            DoG.append(conv)
            scores.append(cv2.resize(conv, (im_width, im_height), interpolation = cv2.INTER_AREA))
            radii.append(2**pow * sigmas[i] * np.sqrt(2))
        DoG = np.stack(DoG)
        min_max.append(sparse_expand(((DoG == ndimage.maximum_filter(DoG, size=5)) | (-DoG == ndimage.maximum_filter(-DoG, size=5))), im_width, im_height, 2**pow))
        pow += 1
    
    #This section of code calculates the minmax kernel. 
    scores = abs(np.stack(scores))
    min_max = np.concatenate(min_max, axis=0)
    blobs = []
    for i in range(0, min_max.shape[0]):
        for j in range(0, min_max.shape[1]):
            for k in range(0, min_max.shape[2]):
                if min_max[i][j][k]:
                    if scores[i][j][k] > threshold:
                        blobs.append((j, k, radii[i], scores[i][j][k]))
    return np.array(blobs)

def sparse_expand(orig, width, height, scale):
    full = np.zeros((orig.shape[0], height, width))
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            for k in range(orig.shape[2]):
                if orig[i][j][k]:
                    full[i][j * scale][k*scale] = 1
    return full

def gaussian(sigma, side_length):
    filter = np.zeros([side_length, side_length], dtype=float)
    middle = side_length / 2 - 0.5
    for i in range(0, side_length):
        for j in range(0, side_length):
            filter[i][j] = (1 / (2 * np.pi * sigma**2)) * np.exp(((i-middle)**2 + (j-middle)**2) / (-2 * sigma**2))
    return filter

"""
arr = gaussian(2)
print('\n'.join([''.join(['{:10}'.format(round(item, 4)) for item in row]) 
      for row in arr]))
"""