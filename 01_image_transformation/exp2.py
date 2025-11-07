# opencv-python numpy matplotlib
# Build a Data Augmentation Pipeline —
# that means: take a few original images, randomly transform them, and generate new versions to expand your dataset.


import cv2
import numpy as np
import os 
import random 
import matplotlib.pyplot as plt

# 1. load sample image
image = cv2.imread(cv2.samples.findFile('image1.jpg'))
if image is None:
    image = np.zeros((200,200,3), dtype=np.uint8)
    cv2.putText(image, 'A.I.', (50,120),  cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

rows, cols = image.shape[:2]   #image height , width in  pixel

# 2. function to apply random transformation 

def random_affine_transformation(img):
    #  random rotation btw -30 & 30 degree
    angle = random.uniform(-30,30)

    # random scaling btw 0.8x - 1.2x
    scale = random.uniform(0.8,1.2)

    # random shear btw -0.3 & 0.3
    shear = random.uniform(-0.3,0.3)

    #  random flip choice 
    flip_horizontal = random.choice([True , False])
    flip_vertical = random.choice([True, False])

    # roatatio & scale matrix
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, scale)

    # shear matrix 
    # convert 2x3 rotation matrix into 3x3 
    M_rot_3x3 = np.vstack([M_rot, [0,0,1]])
    shear_matrix = np.array([[1,shear,0],
                             [shear,1,0],
                             [0,0,1]], dtype=np.float32)
    
    # combine rotation/scale and shear
    M_combined = shear_matrix @ M_rot_3x3

    # convert back to 2x3 for warpaffine
    M_affine = M_combined[:2, :]

    # apply transformation 
    transformed = cv2.warpAffine(img, M_affine, (cols, rows),borderMode=cv2.BORDER_REFLECT)

    # apply random flips 
    if flip_horizontal:
        transformed = cv2.flip(transformed, 1)
    if flip_vertical:
        transformed = cv2.flip(transformed, 0)

    return transformed

# 3. generate augmented dataset 
augmented_images = [random_affine_transformation(image) for _ in range(8)]

# 4. display original and augmented images
plt.figure(figsize=(8,8))
plt.subplot(3,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

for i, aug_img in enumerate(augmented_images):
    plt.subplot(3,3,i+2)
    plt.imshow(cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Aug{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()