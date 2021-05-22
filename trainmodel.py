import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('C:/Users/Sushmita Richhariya/OneDrive/Desktop/New folder (2)/photo.jpg')
#print(img[0])
#print(img)
#print(img.shape)


haar_data = cv2.CascadeClassifier('C:/example/haarcascade_frontalface_default.xml')
print(haar_data.detectMultiScale(img))
cv2.destroyAllWindows()
capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        face = haar_data.detectMultiScale(img)
        for x,y,w,h in face:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            #face = face.reshape(1, -1)
            #face = pca.transform(face)
            #svm = SVC()

            #pred = svm.predict(face)
            #n = name[int(pred)]
            #print(n)
            cv2.imshow("result" ,img)
            print(len(data))
            if len(data) < 400:

                data.append(face)

        if cv2.waitKey(2) == 27 or len(data) >=200:


            break



capture.release()
cv2.destroyAllWindows()
#np.save('without_mask.npy', data)
np.save('with_mask.npy', data)
print(plt.imshow(data[0]))






