import os, sys
import cv2
import numpy as np
from utilsImageStitching import *
from detectBlobs import *

imagePath = sys.argv[1]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn)))

# Build your strategy for multi-image stitching. 
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the 
# number of inliers to each transformation, use this information to determine which 
# pair of images should be merged first (and of these, which one should be the "source" 
# and which the "destination"), merge this pair, and proceed recursively.

# YOUR CODE STARTS HERE

cv2.imshow('ajisdfjioasdfj', images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoints = [detectKeypoints(img) for img in images]
descriptors = [computeDescriptors(img, key) for key,img in zip(keypoints, images)]
H = [[[], [], []], [[], [], []], [[], [], []]]
pairs = []
for i in range(len(images)):
    for j in range(i+1, len(images)):
        matches = getMatches(descriptors[i], descriptors[j])
        H[i][j], inliers = RANSAC(matches, keypoints[i], keypoints[j])
        pairs.append(((i, j), inliers))


print(pairs)   
