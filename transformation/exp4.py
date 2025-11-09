# mini document scanner

import cv2
import numpy as np

# 1. load the image
image = cv2.imread("image1.jpg")  

if image is None:
    raise FileNotFoundError("Image not found")

orig = image.copy()
image = cv2.resize(image, (800, int(image.shape[0] * 800 / image.shape[1])))  # new width, height

points = []

# 2 create window and mouse callback
def get_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Corners", image)

cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", get_points)
cv2.imshow("Select Corners", image)

print("Click four corners of the document (Top-Left → Top-Right → Bottom-Right → Bottom-Left)")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. proceed only if 4 points are selected
if len(points) != 4:
    print("Error: You must select exactly 4 points.")
    exit()

width, height = 600, 800
dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# 4. perspective transform
M = cv2.getPerspectiveTransform(np.float32(points), dst_points)
warped = cv2.warpPerspective(orig, M, (width, height))

# 5. result
cv2.imshow("Original Image", orig)
cv2.imshow("Scanned (Warped) Image", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
