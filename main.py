import numpy as np
from PIL import Image
from math import log2

def parseimg(img):
    pixels = np.array(img.getdata())
    if len(np.shape(pixels)) == 1:
        pixels = pixels[:, None]
    dim = np.shape(pixels)[-1]
    return pixels, dim

def calcpmf(arr, dimension):
    val, cnt = np.unique(arr, return_counts=True, axis=0)
    pmf = cnt / len(arr)
    myarr = np.hstack((val.reshape(-1,dimension),pmf.reshape(-1,1)))
    return myarr

def calcentropy(input):
    entropy = 0
    for i in input:
        entropy -= i[-1]*log2(i[-1])
    return entropy

def jointpmf(arr1, arr2):
    PXY = np.zeros((np.shape(arr1)[0], np.shape(arr2)[0]))
    for i in range(np.shape(arr1)[0]):
        for j in range(np.shape(arr2)[0]):
            PXY[i, j] = arr1[i, -1] * arr2[j, -1]
    return PXY

def mainfunc(img):
    pix, dim = parseimg(img)
    pmf = calcpmf(pix, dim)
    entropy = calcentropy(pmf)
    return entropy

if __name__ == "_main_":
    img_gs = Image.open('../../Downloads/ISAE_Logo_SEIS_gs.png', 'r')
    img_gs_noisy = Image.open('../../Downloads/ISAE_Logo_SEIS_gs_noisy.png', 'r')
    img_clr = Image.open('../../Downloads/ISAE_Logo_SEIS_clr.png', 'r')
    img_clr_noisy = Image.open('../../Downloads/ISAE_Logo_SEIS_clr_noisy.png', 'r')

    print("Entropy GS: ", mainfunc(img_gs))
    print("Entropy GS noisy: ", mainfunc(img_gs_noisy))
    print("Entropy CLR: ", mainfunc(img_clr))
    print("Entropy CLR noisy: ", mainfunc(img_clr_noisy))