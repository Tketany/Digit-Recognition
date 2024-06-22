import cv2 as cv
import numpy as np

pricePerKw=0.1082


img = cv.imread("Board_Images/img4.jpg")
img_resized = cv.resize(img,(300,400))
img_grey = cv.cvtColor(img_resized,cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_grey,(3,3),cv.BORDER_DEFAULT) #filtering from noises, remove minor edges
#cv.imshow("img_blur",img_blur)
img_canny = cv.Canny(img_blur,125,175)
contours,hierarchies= cv.findContours(img_canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE) #cv.RETR_LIST lists all contours
#print(contours)

#sort contours based on area 
cont=sorted(contours,key=cv.contourArea,reverse=True)[0]

#draw the contours on the resized image, -1 to draw all contours, color green, 1 is thickness
cv.drawContours(img_resized, cont, -1, (0, 255, 0), 1) 
#cv.imshow("img",img_resized)

x,y,w,h=cv.boundingRect(cont)
#print(x,y,w,h)
regionOfInterest=img_blur[y:y+h,x:x+w]

#cv.imshow("board",regionOfInterest)

#save image
cv.imwrite("Boards/board1.png",regionOfInterest)

#slicing into 1/6 portions
nums=[]
for i in range(1,7):
    num=img_blur[y:y+h,int(x+w*(i-1)/6):int(x+w*i/6)]
    #imgname="img"+str(i)
    #print(imgname)
    #cv.imshow(imgname,num)
    nums.append(num)
    


#predict number using the knn

import predictNum
#d0_resized = cv.resize(nums[3],(28,28))
print("------------------")
#ret,d0_thresh=cv.threshold(d0_resized,50,255,cv.THRESH_BINARY)
#cv.imshow("d0_resizedh",d0_resized)
#cv.imshow("d0_threshh",d0_thresh)
#filtering top,bottom,right and left from noises
#d0_thresh[0]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[1]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[2]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[3]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[4]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[5]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[24]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[25]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[26]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#d0_thresh[27]=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#for i in range(0,28):
#    d0_thresh[i][0]=0
#    d0_thresh[i][1]=0
#    d0_thresh[i][27]=0
#    d0_thresh[i][26]=0

#d0_dlt = cv.dilate(d0_thresh, np.ones((2, 2), np.uint8), iterations=1)
#d0_blur = cv.GaussianBlur(d0_dlt,(3,3),cv.BORDER_DEFAULT)
#cv.imshow("d0",d0_blur)
#predictNum.predict(d0_blur)






#read numbers using pyteseract
import pyteseract_Final
boardReadings=pyteseract_Final.main('Boards/board1.png')
print("Board Reading:",boardReadings)
consumption=0
for digitNum in range(0,6):
    if boardReadings[digitNum].isdigit():
        #print((digitNum-5)*-1,boardReadings[digitNum])
        consumption = consumption + (10**((digitNum-5)*-1))*int(boardReadings[digitNum])
consumption=consumption/10
price=consumption*pricePerKw
print("You should pay ",round(price,3),"$")

    

#cv.waitKey(0)