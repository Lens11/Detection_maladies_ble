import cv2
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import exposure

def histogramOpenCV(img):
    _, axis = plt.subplots(ncols=2, figsize=(12, 3))
    axis[0].imshow(img)
    axis[1].set_title('Histogram')
    axis[0].set_title('Image')
    rgbcolors = ['red', 'green', 'blue']
    for i, col in enumerate(rgbcolors):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        axis[1].plot(histr, color=col)
    plt.show()

img = imread('part_0.png')
# Convertir l'image en format attendu par OpenCV (valeurs de 0 à 255)
img = (img * 255).astype('uint8')
histogramOpenCV(img)
