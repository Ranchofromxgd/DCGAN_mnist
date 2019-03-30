import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from PIL import Image

def corelation(img1,img2):
    img1 = img1.reshape(-1)
    img2 = img2.reshape(-1)
    return np.corrcoef(img1, img2)[0, 1]

if __name__ == '__main__':

    img1 = np.array(Image.open('0.png'))
    img2 = np.array(Image.open('2.png'))
    print(corelation(img1,img2))
