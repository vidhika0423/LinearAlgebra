# random experiment 
# 1. load image 
# 2. apply transformation 
# 3. print

import cv2
import numpy as np 
from PIL import Image

# 1
image = cv2.imread(cv2.samples.findFile("image1.jpg"))

rows, cols = image.shape[:2]

# a, d = scaling and rotation
# b.c = shearing
# tx, ty = translation 

a=1.5
d=1.5
b=0
c=-0.5
tx=0
ty=0

m = np.float32([[a,b,tx],
                [c,d,ty]])

transformed = cv2.warpAffine(image, m, (cols, rows))

# Convert from BGR → RGB
transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

# Use Pillow to show image directly
Image.fromarray(transformed_rgb).show()