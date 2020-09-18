import os, sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utilsImageStitching import *
from detectBlobs import *

imagePath = sys.argv[1]
output_path = sys.argv[2]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE))

# Build your strategy for multi-image stitching. 
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the 
# number of inliers to each transformation, use this information to determine which 
# pair of images should be merged first (and of these, which one should be the "source" 
# and which the "destination"), merge this pair, and proceed recursively.

# YOUR CODE STARTS HERE

keypoints = [detectKeypoints(img) for img in images]
descriptors = [computeDescriptors(img, key) for key,img in zip(keypoints, images)]
match_matrix = [[[] for i in range(len(images))] for i in range(len(images))]
pairs = []
for i in range(len(images)):
    for j in range(i+1, len(images)):
        matches = getMatches(descriptors[i], descriptors[j])
        match_matrix[i][j], inliers = RANSAC(matches, keypoints[i], keypoints[j])
        match_matrix[j][i] = np.linalg.inv(match_matrix[i][j])
        pairs.append(((i, j), inliers))

pairs = sorted(pairs, key = lambda x: -x[1])
# Ya boi's getting fancy with Kruskals
parent = [i for i in range(len(images))]

def root(a):
    global parent
    if parent[a] == a:
        return a
    parent[a] = root(parent[a])
    return parent[a]
def union(a, b):
    parent[root(a)] = root(b)

parent = [i for i in range(len(images))]
H_list = [np.eye(3) for i in range(len(images))]
mask = [0 for i in range(len(images))]

for pair in pairs:
    a = pair[0][0]
    b = pair[0][1]
    if root(a) != root(b):
        if(pair[1] < 50):
            break
        if(mask[a]):
            c = b
            b = a
            a = c
        union(a, b)
        H_list[a] = np.matmul(H_list[b], match_matrix[a][b])
        mask[a] = 1
        mask[b] = 1
images = [x for x, y in zip(images, mask) if y]
H_list = [x for x, y in zip(H_list, mask) if y]
output = warpImagesWithMapping(images, H_list)
cv2.imwrite(output_path, output)