from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import cv2 as cv
import numpy as np
import os
import time
from sklearn import metrics
import matplotlib.pyplot as plt


#metrics for confusion matrox



def findAccuracy(knn):
    testset=[]
    i=0
    while i<=9:
        imgs_test_names=os.listdir('testingSet2/Testingset/'+str(i))
        for img_test_name in imgs_test_names:
            path='testingSet2/Testingset/'+str(i)+'/'+str(img_test_name)
            img_test=cv.imread(path)
            img_test=cv.cvtColor(img_test,cv.COLOR_BGR2GRAY)
            img_test=img_test.flatten()
            testset.append(img_test)
        i=i+1
    correctLabels=[]
    for x in range(0,10):
        for label in range(0,len(os.listdir('testingSet2/Testingset/'+str(x)))):
            correctLabels.append(x)
    
    predictedLabels=[]
    for test_img in testset:
        y_pred_test = knn.predict([test_img])
        predictedLabels.append(y_pred_test[0])
    #print(predictedLabels)
    errorcnt=0
    if len(predictedLabels) == len(correctLabels): #make sure that the lengths are the same
        for m in range(0,len(predictedLabels)):
            if predictedLabels[m] != correctLabels[m]:
                errorcnt=errorcnt+1
    confusionMatrix=metrics.ConfusionMatrixDisplay.from_predictions(correctLabels,predictedLabels)
    confusionMatrix.figure_.suptitle("Confusion Matrix")
    #print(confusionMatrix.confusion_matrix)
    plt.show()
    return (1-(errorcnt/len(predictedLabels)))
        

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (K) as needed


imgs=[]
i=0
while i<=9:
    imgs_names=os.listdir('Trainingset_sample/'+str(i))
    for img_name in imgs_names:
        path='Trainingset_sample/'+str(i)+'/'+str(img_name)
        img=cv.imread(path)
        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img=img.flatten()
        imgs.append(img)
    i=i+1

y_train=[]
for i in range(0,10):
    for j in range(0,750):
        y_train.append(i)
#print(y_train)

# Train the classifier on the training data
t1=time.time()
knn.fit(imgs, y_train)





img = cv.imread("Testingset_sample/img_73.jpg", cv.IMREAD_GRAYSCALE)  # Read the image in grayscale


#cv.imshow("imgmgmg",resized_img)
feature_vector = img.flatten()

# Flatten the resized image to create a feature vector
y_pred = knn.predict([feature_vector])
# Print the predicted label
# y_pred is array that has 1 element: predicted value
t2=time.time()
print("Predicted digit:", y_pred[0])
print("Time taken to train and predict:",(t2-t1))



#metrics
print("Accuracy:",findAccuracy(knn))


#cv.waitKey(0)