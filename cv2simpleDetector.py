# import the necessary packages
from __future__ import print_function
import imutils
import cv2
# load the Tetris block image, convert it to grayscale, and threshold
# the image
print("OpenCV Version: {}".format(cv2.__version__))
image = cv2.imread("C:/Users/Gerardo/Documents/PythonData/wp4.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)

# draw the contours on the image
cv2.drawContours(image, cnts, -1, (240, 0, 159), 3)  # -1 es plotear todo, color en (r,g,b), y tamaño de la línea
cv2.imshow("Image", image)
cv2.waitKey(0)
