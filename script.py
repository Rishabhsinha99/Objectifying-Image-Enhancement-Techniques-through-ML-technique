

from skimage import io,color,exposure,filters
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
    

#%%

files=os.listdir("C:\\Users\\asus\\Desktop\\DIP\\Pict20")

arr=[]    
for i in files:
    img = io.imread(i)
    gray=color.rgb2gray(img)
    hist=plt.hist(gray.ravel(),bins=256)
    mm_sc=MinMaxScaler()
    freq=hist[0]
    freq=freq.reshape(-1,1)
    freq_mm=mm_sc.fit_transform(freq)
    new=freq_mm.reshape(1,256)
    lt=[i for i in new[0]]
    arr.append(lt)

X=np.array(arr)
 
from sklearn.cluster import KMeans
wcss= []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.show()

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)

#%%

A=[i for i in range(len(y_kmeans)) if y_kmeans[i]==0]
B=[i for i in range(len(y_kmeans)) if y_kmeans[i]==1]
C=[i for i in range(len(y_kmeans)) if y_kmeans[i]==2]
D=[i for i in range(len(y_kmeans)) if y_kmeans[i]==3]
# E=[i for i in range(len(y_kmeans)) if y_kmeans[i]==4]
# F=[i for i in range(len(y_kmeans)) if y_kmeans[i]==5]

#%%
def read(file):
    img = io.imread(file)
    gray=color.rgb2gray(img)
    #io.imshow(gray)

#histogram original image
    hist=plt.hist(gray.ravel(),bins=256)
    plt.title(file+' Histogram')
    #plt.savefig("C:\\Users\\asus\\Desktop\\"+file+".jpg")
    plt.show(hist)

#%%
for q in A:
    read(files[q])
    
#%%

img = io.imread('5.jpg')
gray=color.rgb2gray(img)
io.imshow(gray)

#histogram original image
hist=plt.hist(gray.ravel(),bins=256)
plt.show(hist)

#histogram Equalization
eq_img=exposure.equalize_hist(gray)
eq_hist=plt.hist(eq_img.ravel(),bins=256)
plt.show(eq_hist)
io.imshow(eq_img)

#Contrast Stretching
p0, p200 = np.percentile(gray, (0, 98))
img_rescale = exposure.rescale_intensity(gray, in_range=(p0, p200))
io.imshow(img_rescale)
cont_hist=plt.hist(img_rescale.ravel(),bins=256)
plt.show(cont_hist)


#Adaptive Histogram Equalization (CLAHE)
img_adapteq = exposure.equalize_adapthist(gray, clip_limit=0.03)
io.imshow(img_adapteq)
adapt_hist=plt.hist(img_adapteq.ravel(),bins=256)
plt.show(adapt_hist)

    
#Unsharp Masking
unsharp=filters.unsharp_mask(gray,200,1)
io.imshow(unsharp)
unsh_hist=plt.hist(unsharp.ravel(),bins=256)
plt.show(unsh_hist)

#log
log=exposure.adjust_log(gray,1)
io.imshow(log)
log_hist=plt.hist(log.ravel(),bins=256)
plt.show(log_hist)

#Gamma
gamma=exposure.adjust_gamma(gray,0.4)
io.imshow(gamma)
gamma_hist=plt.hist(gamma.ravel(),bins=256)
plt.show(gamma_hist)

#median 
median=filters.median(eq_img)
io.imshow(median)
    


