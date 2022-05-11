import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import rasterio
from rasterio import plot
import PIL
from PIL import Image
import imageio
import numpy as np
import os

os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'

# image_path = "/SENTINEL-2/GRANULE/L2A_T44RNP_A034765_20220217T050856/IMG_DATA/R20m"

image_2021_b6 = "SENTINEL-2-(2021-10-10)/GRANULE/L2A_T44RNP_A032906_20211010T052120/IMG_DATA/R60m/T44RNP_20211010T050731_B06_60m.jp2"
# image_2021_b7 = "T44RNP_20211010T050731_B07_20m.jp2"
# image_2021_b8a = "T44RNP_20211010T050731_B8A_20m.jp2"

image_2022_b6 = "SENTINEL-2-(2022-04-28)/GRANULE/L2A_T44RNP_A035766_20220428T051943/IMG_DATA/R60m/T44RNP_20220428T050701_B06_60m.jp2"
# image_2022_b7 = "T44RNP_20220428T050701_B07_20m.jp2"
# image_2022_b8a = "T44RNP_20220428T050701_B8A_20m.jp2"

band1_2021 = rasterio.open(image_2021_b6, driver = 'JP2OpenJPEG')
# band2_2021 = rasterio.open(image_2021_b7, driver = 'JP2OpenJPEG')
# band3_2021 = rasterio.open(image_2021_b8a, driver = 'JP2OpenJPEG')

band1_2022 = rasterio.open(image_2022_b6, driver = 'JP2OpenJPEG')
# band2_2022 = rasterio.open(image_2022_b7, driver = 'JP2OpenJPEG')
# band3_2022 = rasterio.open(image_2022_b8a, driver = 'JP2OpenJPEG')

print(band1_2021.count)
print(band1_2021.width) # number of columns
print(band1_2021.height) # number of rows
print(band1_2021.dtypes) # data type of band
print(band1_2021.crs) # crs
print(band1_2021.transform) # transform parameters of band

plot.show(band1_2021)
plot.show(band1_2022)

print(band1_2022.count)
print(band1_2022.width) # number of columns
print(band1_2022.height) # number of rows
print(band1_2022.dtypes) # data type of band
print(band1_2022.crs) # crs
print(band1_2022.transform) # transform parameters of band

im1 = Image.open(r'SENTINEL-2-(2021-10-10)/GRANULE/L2A_T44RNP_A032906_20211010T052120/IMG_DATA/R20m/T44RNP_20211010T050731_B06_20m.jp2')
im1.save(r'image1.png')
im2 = Image.open(r'SENTINEL-2-(2022-04-28)/GRANULE/L2A_T44RNP_A035766_20220428T051943/IMG_DATA/R20m/T44RNP_20220428T050701_B06_20m.jp2')
im2.save(r'image2.png')

imagepath1 = 'image1.png'
imagepath2 = 'image2.png'

image1 = imageio.imread(imagepath1)
image2 = imageio.imread(imagepath2)
image1 = image1.astype(np.int16)
image2 = image2.astype(np.int16)
diff_image = abs(image1 - image2)
print(diff_image)

print(diff_image.shape)

data = Image.fromarray(diff_image)
# saving the final output 
# as a PNG file
data.save('difference_image.png')

def find_vector_set(diff_image, new_size):
 
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
 
    mean_vec   = np.mean(vector_set, axis = 0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):
 
    i = 2
    feature_vector_set = []
 
    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
 
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print("feature vector space size", FVS.shape)
    return FVS

def clustering(FVS, components, new):
 
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)
 
    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    return least_index, change_map

def find_PCAKmeans(imagepath1, imagepath2):
 
    image1 = imageio.imread(imagepath1)
    image2 = imageio.imread(imagepath2)
 
    new_size = np.asarray(image1.shape) / 5
    new_size = new_size.astype(int) * 5
    # image1 = imresize(image1, (new_size)).astype(np.int16)
    # image2 = imresize(image2, (new_size)).astype(np.int16)
    im1 = Image.fromarray(image1)
    image1 = np.array(im1.resize(new_size, PIL.Image.BICUBIC))
    im2 = Image.fromarray(image2)
    image2 = np.array(im2.resize(new_size, PIL.Image.BICUBIC))
 
    diff_image = abs(image1 - image2)
    imageio.imsave('diff.jpg', diff_image)
 
    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_
 
    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)
 
    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0
 
    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    imageio.imsave("changemap.jpg", change_map)
    imageio.imsave("cleanchangemap.jpg", cleanChangeMap)

a = 'image1.png'
b = 'image2.png'
find_PCAKmeans(a,b)