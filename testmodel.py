import numpy as np
import cv2


from sklearn.decomposition import PCA


without_mask = np.load('without_mask.npy')
with_mask = np.load('with_mask.npy')
print(with_mask.shape)
print(without_mask.shape)
with_mask = with_mask.reshape(200, 50*50*3)

without_mask = without_mask.reshape(200, 50*50*3)
print(with_mask.shape)
print(without_mask.shape)


x = np.r_[with_mask, without_mask]
x.shape
labels = np.zeros(x.shape[0])
labels[200:] = 1.0
name = {0 : 'Mask', 1 : 'No_Mask'}
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.20)

from sklearn.decomposition import PCA




pca =PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_train[0]
print(x_train.shape)

svm = SVC()
svm.fit(x_train, y_train)
x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)
print( accuracy_score(y_test, y_pred))
img = cv2.imread('C:/Users/Sushmita Richhariya/OneDrive/Desktop/New folder (2)/photo.jpg')
#print(img[0])
#print(img)
#print(img.shape)


haar_data = cv2.CascadeClassifier('C:/example/haarcascade_frontalface_default.xml')
#print(haar_data.detectMultiScale(img))

capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            face = face.reshape(1, -1)
            face = pca.transform(face)


            pred = svm.predict(face)
            n = name[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,230,250), 2)
            print(n)
            cv2.imshow("result" ,img)
           # print(len(data))
            #if len(data) < 200:

               # data.append(face)

        if cv2.waitKey(2) == 27:# or len(data) >=200:


            break
capture.release()
cv2.destroyAllWindows()
#np.save('without_mask.npy', data)
#np.save('with_mask.npy', data)



