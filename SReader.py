
###################################################################
####     Smart Reader for Visually Impaired People             ####
####               2021 Graduation Project                     ####
#### Muhammed Samo, Furkan Coşkuner, Uğur Çankaya, Ömer Yavuz  ####
###################################################################

import os
import math
import cv2
import pytesseract
import numpy as np
import re

from matplotlib import pyplot as plt
from typing import Tuple, Union
from deskew import determine_skew
from gtts import gTTS
from langdetect import detect

def rotate(
        image: np.ndarray, angle: float, background: \
        Union[int, Tuple[int, int, int]]) -> np.ndarray:
    
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + \
            abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + \
             abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), \
                                           int(round(width))), \
                          borderValue=background)

###################################################################

pytesseract.pytesseract.tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract.exe'
url = "http://192.168.1.39:8080/video"
kamera = cv2.VideoCapture(url)

############# -- Video Capture -- ##################################

while True:
    ret,goruntu=kamera.read()
    ############## -- Resize Captured Video -- #####################
    scale_percent = 32 
    width = int(goruntu.shape[1] * scale_percent / 100)
    height = int(goruntu.shape[0] * scale_percent / 100)
    dim = (width, height)
    goruntu = cv2.resize(goruntu, \
                         dim, interpolation = cv2.INTER_AREA)
    ################################################################
    cv2.imshow("Live Video",goruntu)
    
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.imwrite("1Captured.jpg",goruntu)
        break

kamera.release()
cv2.destroyAllWindows()

################# -- Captured Image -- #############################
    
captured = cv2.imread("1Captured.jpg")   
cv2.imshow("Captured Image",captured)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        #cv2.destroyAllWindows()
        break
    
################ -- Rotated Image -- ###############################    

rotated = cv2.rotate(captured, cv2.cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated Image",rotated)
cv2.imwrite("2Rotated.jpg",rotated)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.destroyAllWindows()
        break
    
################## -- BGR to Gray -- ###############################
    
bgr2gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
cv2.imshow("Rotated  Image",rotated)
cv2.imshow("BGR to Gray",bgr2gray)
cv2.imwrite("3Gray.jpg",bgr2gray)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.destroyAllWindows()
        break
    
################# -- Thicken Letters -- ############################
    
kernel = np.ones((2,2),np.uint8)
eroded = cv2.erode(bgr2gray,kernel,iterations = 1)
cv2.imshow("BGR to Gray",bgr2gray)
cv2.imshow("Thicken Letters",eroded)
cv2.imwrite("4Thicken.jpg",eroded)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.destroyAllWindows()
        break
    
################ -- Remove Shadows--  ##############################

rgb_planes = cv2.split(eroded)

result_planes = []
result_norm_planes = []

for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0,\
                             beta=255, norm_type=cv2.NORM_MINMAX,\
                             dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv2.merge(result_planes)
result_norm = cv2.merge(result_norm_planes)
#cv2.imshow("Removed Shadows1",result)

cv2.imshow("Thicken Letters",eroded)
cv2.imshow("Removed Shadows2",result_norm)
cv2.imwrite("5Shadows.jpg",result_norm)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.destroyAllWindows()
        break
    
################### -- De-Skewing -- ###############################
  
osd_rotated_image=pytesseract.image_to_osd(result_norm)
angle_rotated_image = re.search('(?<=Rotate: )\d+',\
                                osd_rotated_image).group(0)

#lng_ımg = re.search('(?<=Script: )\d+',osd_rotated_image).group(0)
print(osd_rotated_image)

#if( lng_ımg == 'Arabic'):
#    result_norm = cv2.rotate(result_norm,cv2.ROTATE_180)

    
if (angle_rotated_image == '0'):
    img = cv2.rotate(result_norm,cv2.ROTATE_180)
    angle = determine_skew(img)
    angled = rotate(img, angle, (0, 0, 0))   
else:
    angle = determine_skew(result_norm)
    angled = rotate(result_norm, angle, (0, 0, 0))

cv2.imshow("Threshold Method",result_norm)
cv2.imshow("Rotated with Angle",angled)
cv2.imwrite("6Angled.jpg",angled)

while True:
    if cv2.waitKey(30) & 0xFF == ord('w'):
        cv2.destroyAllWindows()
        break
    
####################################################################
    
titles  = ['1.Captured Image','2.BGR to Gray','3.Thicken Letters',
           '4.Removed Shadows','5. Rotated Image']
images  = [rotated,bgr2gray,eroded,result_norm,angled]

for i in range(5):
    plt.subplot(1,6,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

################### -- Image to Text -- ############################

conf = r'-l eng+tur --oem 3 --psm 6'
txt = pytesseract.image_to_string(angled, config=conf)
print(txt)      
lng=detect(txt)


################### -- Text to Speech -- ###########################
          
out = gTTS(text=txt,slow=True,lang=lng)
out.save('out.mp3')
os.system("start out.mp3")
cv2.imshow("Rotated with Angle",angled)

####################################################################

















    

