#affine transformation

import cv2
import numpy as np

# 1. load image
image = cv2.imread(cv2.samples.findFile("image1.jpg"))
if image is None:
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(image, (200, 200), 80, (0,255,0), -1)
    cv2.putText(image, 'A.I.', (150,210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

rows, cols = image.shape[:2]
# 2. function to apply affine transformation 

def apply_transform():
    a = cv2.getTrackbarPos('a', 'Transformer')/100.0
    b = cv2.getTrackbarPos('b', 'Transformer')/100.0
    c = cv2.getTrackbarPos('c', 'Transformer')/100.0
    d = cv2.getTrackbarPos('d', 'Transformer')/100.0
    tx = cv2.getTrackbarPos('tx', 'Transformer') -200
    ty = cv2.getTrackbarPos('ty', 'Transformer')-200

    # a= 1.5
    # d=0.5
    # b=0.5
    # c=0.5
    # tx=250
    # ty=230

    #construct 2 x 3 affine matrix
    M= np.float32([[a, b, tx],[c, d, ty]])

    #apply affine transformation
    transformed = cv2.warpAffine(image, M, (cols,rows))
    cv2.imshow('Transformed Image', transformed)

# 3. create window and trackbar
cv2.namedWindow('Transformer')
cv2.namedWindow('Transformed Image')

cv2.createTrackbar('a', 'Transformer', 100,200 , lambda x: apply_transform())
cv2.createTrackbar('b', 'Transformer', 0 , 200, lambda x: apply_transform())
cv2.createTrackbar('c', 'Transformer', 0 , 200, lambda x:apply_transform())
cv2.createTrackbar('d', 'Transformer', 100, 200, lambda x:apply_transform())
cv2.createTrackbar('tx', 'Transformer', 200,400, lambda x: apply_transform())
cv2.createTrackbar('ty' , 'Transformer', 200,400, lambda x: apply_transform())

#initial call
apply_transform()

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break # esc key to quit
    apply_transform()

cv2.destroyAllWindows()


"""
affine transformation 
a d= scaling & rotation 
b c= shearing & rotation 
tx ty = translation (left/ right/ up / down)
"""
