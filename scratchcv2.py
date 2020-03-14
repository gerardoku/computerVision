# Parasites and Vectors 3.031
# PlosONE 2.776
# Parasitology Research 2.067
# Physiological and Biochemical Zoology 1.873

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Trabajos/Manuscritos/Maritere/data/smears/Day_5/1PY.R/104_p1.bmp')
plt.imshow(img, 'gray')
plt.show()
img = cv2.medianBlur(img,5)
plt.imshow(img, 'gray')
plt.show()

# plot channels
b,g,r = cv2.split(img)
print(b, g, r)
images = [b, g, r]
titles = ['b', 'g', 'r']
for i in range(0, 3):
    plt.subplot(1,3,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

plt.imshow(r,'gray')
plt.title('red')
plt.show()

# gray
img = cv2.imread('C:/Trabajos/Manuscritos/Maritere/data/smears/Day_5/1PY.R/104_p1.bmp', 0)
img = cv2.medianBlur(img,5)

# Calculae thresholds
th = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY_INV)[1]
ret, th1 = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Experiment
new = th + th1
#print(img, th, th1, th2, th3, new)

titles = ['img: Original',
			'th: Global Thresholding INV (v = 235)', 
			'th1: Global Thresholding (v = 145)',
            'th2: Adaptive Mean Thresholding', 
            'th3: Adaptive Gaussian Thresholding',
            'new: Experiment (see source code)']
images = [img, th, th1, th2, th3, new]

# Plots thresholds
for i in range(0, 6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
#plt.show()

# Contours ------------------------------------

contours,hierarchy = cv2.findContours(th1, 1, 2)
# print(len(contours))  # 266
for i in range(0, len(contours) - 1):
	cnt = contours[i]
	area = cv2.contourArea(cnt) 
	if area > 500 and area < 2000:
		#M = cv2.moments(cnt)
		ellipse = cv2.fitEllipse(cnt)
		im = cv2.ellipse(img, ellipse, (0,255,0), 2)


# -------------------------------------
# Blob detector para th1

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
#params.minThreshold = 50  # Darker (lightness como en HSL)
#params.maxThreshold = 100  # Lighter

# Filter by Area.
params.filterByArea = True
params.minArea = 500
params.maxArea = 3000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0  # Super sensible en valores altos
params.maxCircularity = 0.85  # Super sensible en valores altos

# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
#params.maxInertiaRatio = 0.9

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
#params.maxConvexity = 1

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(th1)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
