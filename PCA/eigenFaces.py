import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

# 1. load face dataset

faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data      #image is flattened into !D array
y = faces.target    # labels (person name)
images = faces.images   #original 2D images

# print("Dataset shape:", X.shape)
# print("each img shape:", images[0].shape)
# Dataset shape: (1348, 2914)
# each img shape: (62, 47)

# 2. applying pca to find main face features
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(X)

# print("Explained variance ratio:", np.sum(pca.explained_variance_ratio_))
# Explained variance ratio: 0.9039648
# Variance ratio: how much original image information is retained

# 3. top 10 eigenfaces

# fig, axes= plt.subplots(2,5, figsize=(10,5),
#                         subplot_kw={'xticks':[],'yticks':[]})

# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(images[0].shape), cmap='gray')
#     ax.set_title(f"PC {i+1}")

# plt.suptitle("Top 10 Eigenfaces ")
# plt.show()

# 4. reconstruct faces using fewer components

X_pca = pca.transform(X)        #original data into PCA space i.e original to PCA
X_reconstructed = pca.inverse_transform(X_pca)          # pca data back tp original data 

fig, axes = plt.subplots(2,10, figsize=(15,4),
                         subplot_kw={'xticks':[], 'yticks':[]})

for i in range(10):
    # original 
    axes[0,i].imshow(images[i], cmap='gray')
    axes[0,i].set_title("original")

    # reconstruct
    axes[1,i].imshow(X_reconstructed[i].reshape(images[0].shape), cmap="gray")
    axes[1, i].set_title("Reconstructed")

plt.suptitle("Original vs Reconstructed Faces using PCA (50 components)")
plt.show()