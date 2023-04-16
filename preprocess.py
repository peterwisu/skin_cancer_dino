import cv2
from PIL import Image
from glob import glob
import os
from tqdm import tqdm
import numpy as np

# Hair Removal Algorithms
def dull_razor(image):
    image = np.array(image)
    # covert image to gray scale 
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Black hat filter
    kernel = cv2.getStructuringElement(1,(9,9)) 
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    # Gaussian filter
    gaussian_filter = cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    #Binary thresholding (MASK)
    _,mask = cv2.threshold(gaussian_filter,10,255,cv2.THRESH_BINARY)
    #Replace pixels of the mask
    result = cv2.inpaint(image,mask,6,cv2.INPAINT_TELEA) 
    result= Image.fromarray(result)
    return result

def main():

    all_images = glob(os.path.join("ISIC_2019_Training_Input/",'*.jpg'))

    bar = tqdm(all_images)

    for name in bar:
        
        img =  Image.open(name)

        img = dull_razor(img)

        save_name = os.path.split(name)[1]

        img.save(os.path.join('prepro_data',save_name))


main()



