import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import cv2 as cv



def getBinaryImage(image):
    I = cv.imread(image, 0)
    H = np.zeros(256)
    for i in range(256):
        H[i]=np.sum(I==i)   # histogram

    TotalEntropy=[]

    for T in range(256):
        sum=0
        for j in range(T):
            sum=sum+H[j]

        ptminus,ptplus=[],[]
        for x in range(T):
            if (sum == 0 or H[x]==0):
                continue
            else:
                pT = H[x] / sum  #gray level below threshold
            ptminus.append(pT)


        for x in range(T, 256):
            if ((np.sum(H) - sum )== 0 or H[x]==0):
                continue
            else:
                pT = H[x] / (np.sum(H) - sum)  #gray level above threshold
            ptplus.append(pT)


        Ha,Hb=0,0

        for i in ptminus:
            Ha=Ha-i*np.log2(i)   # Calculating entropy

        for i in ptplus:
            Hb=Hb-i*np.log2(i)   # Calculating entropy


        TotalEntropy.append(Ha+Hb)

    maxthres=max(TotalEntropy)

    indexThreshold=TotalEntropy.index(maxthres) #taking out the intensity at maximum entropy

    final_image = np.copy(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if (final_image[i, j] >= indexThreshold):   # if intenstiy is greater than intensity at maximum threshold than marking it as 1
                final_image[i, j] = 1
            else:
                final_image[i, j] = 0
    return final_image

# Plotting all images
def plot_image(img1,title1,img2,title2):
    I = cv.imread(img1, 0)
    plt.subplot(211)
    plt.imshow(I,cmap=cm.gray)
    plt.title(title1)


    plt.subplot(212)
    plt.imshow(img2, cmap=cm.gray)
    plt.title(title2)
    plt.show()

# output images are present in output folder for all three images

output=getBinaryImage("question2_1.jpg")
plot_image("question2_1.jpg","Original Gray scale image",output,"Binary image")

output=getBinaryImage("question2_2.jpg")
plot_image("question2_2.jpg","Original Gray scale image",output,"Binary image")

output=getBinaryImage("question2_3.jpg")
plot_image("question2_3.jpg","Original Gray scale image",output,"Binary image")


