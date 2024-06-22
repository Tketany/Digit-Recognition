import cv2 as cv
import os
import time
import statistics as st
import numpy as np

def prepareEmptyAvgMatrix():
    matr=[]
    i=0
    j=0
    while i<28:
        matr.append([])
        while j<28:
            matr[i].append([])
            j=j+1
        i=i+1
        j=0
    return matr

def findAverageImages(imgs):
    avgMatrices=[]
    for digit in imgs:
        avgMatrix=prepareEmptyAvgMatrix()
        for px_x in range(0,28):
            for px_y in range(0,28):
                a=[]
                for ele in digit:
                    a.append(ele[px_x][px_y])
                avgMatrix[px_x][px_y].append(st.mean(a))
        #convert the matrix from 3d to 2d to follow same format
        flattened_avgMatrix = np.array(avgMatrix).reshape(28,28)
        #avgMatrices.append(avgMatrix)
        avgMatrices.append(flattened_avgMatrix)
    return avgMatrices

    
def findkminPred(comp,k):
    a=[]
    for i in range(0,k):
        print(min(comp)," /// ",comp.index(min(comp)))
        a.append(int(comp.index(min(comp))/250))
        comp.remove(min(comp))
    return a



imgs=[[],[],[],[],[],[],[],[],[],[]]
i=0
while i<=9:
    imgs_names=os.listdir('Trainingset_sample/'+str(i))
    for img_name in imgs_names:
        path='Trainingset_sample/'+str(i)+'/'+str(img_name)
        img=cv.imread(path)
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        imgs[i].append(img)
    i=i+1

avgMatrices=findAverageImages(imgs)

#print each avg matrix
for m in avgMatrices:
    print(m)
    print("**************************************************")
#show average images
#cv.imshow("0",imgs[9][240])

#img_input=cv.imread('Testingset_sample/img_27.jpg')
img_input=cv.imread('Trainingset_sample/7/img_6.jpg')
img_input=cv.cvtColor(img_input,cv.COLOR_BGR2GRAY)
img_input = cv.resize(img_input, (28, 28))
comp=[]
"""
for mat in avgMatrices:
    s=0
    for px_x in range(0,28):
        for px_y in range(0,28):
            px_value=img_input[px_x][px_y]
            avgMat_value=mat[px_x][px_y]
            #s=s+float(((px_value-avgMat_value)*(px_value-avgMat_value)))
            s=s+abs((px_value-avgMat_value))
    comp.append(s)
print(comp)
minimumComp=comp[0]
predictedLabel=0
for i in range(0,len(comp)):
    if comp[i]<minimumComp:
        minimumComp=comp[i]
        predictedLabel=i

print(predictedLabel)

"""
s=0
for digit in imgs:
    for mat in digit:
        s=0
        for px_x in range(0,28):
            for px_y in range(0,28):
                s=s+(abs(img_input[px_x][px_y]-mat[px_x][px_y]))
        comp.append(s)
print(comp)

print(findkminPred(comp,10))


#print(img_input)
cv.waitKey(0)