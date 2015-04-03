import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm

def get_vessels(img,side):
#convert to grayscale
    #img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #equilized = cv2.equalizeHist(img_gray)
    green_channel = img[:,:,1]
    threshold =np.max(green_channel)*0.9
    
    #crop image
    gch_crop1=green_channel[:, (green_channel != 0).sum(axis=0) != 0]
    gch_crop2=gch_crop1[(gch_crop1 != 0).sum(axis=1) != 0,:]
    green_channel=gch_crop2
    
    #rotate by optical disc
    dummy,gch_bin = cv2.threshold(green_channel, threshold,255 ,cv2.THRESH_BINARY)
    i,j = np.unravel_index(gch_bin.argmax(), gch_bin.shape)
    if ((gch_bin.shape[1]/2 < j) and side=='left') or ((gch_bin.shape[1]/2 > j) and side=='right'):
        green_channel=np.rot90(green_channel,2)
#25 x 25 median filter
    gch_mf = cv2.medianBlur(green_channel,35)
#gch_nl = cv2.fastNlMeansDenoising(green_channel,h=10)
    gch_norm = green_channel - gch_mf

    gch_norm_norm = cv2.medianBlur(gch_norm,35)
#convert to binary image
    thresh,gch_norm_bin = cv2.threshold(gch_norm,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    gch_norm_bin_norm = cv2.medianBlur(gch_norm_bin,35)
    return gch_norm_bin_norm

img=cv2.imread("sample/13_right.jpeg")
get_vessels(img,'right')


#plt.imshow(gch_norm_bin_norm, cmap="gray")
#plt.show()
#sobelx = cv2.Sobel(gch_norm_bin,-1,1,0,ksize=5)
#sobely = cv2.Sobel(gch_norm_bin,-1,0,1,ksize=5)
#laplacian=cv2.Laplacian(gch_norm_bin,-1)

#edges = cv2.Canny(gch_norm,2,255)
#plt.imshow(sobely, cmap="gray")
#plt.show()
#plt.imshow(sobelx, cmap="gray")
#plt.show()

