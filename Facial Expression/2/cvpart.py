import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt


with open('/home/jak/Desktop/DL/Facial Expression/face2', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
img=frame
#img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
img1 = img[y:y+h, x:x+w]

plt.imshow(img1)        #many attributes
plt.show()

def resize2SquareKeepingAspectRation(img, interpolation, size=48):
    h, w = img.shape[:2]
    c = 1 if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)

img1 = resize2SquareKeepingAspectRation(img1, cv2.INTER_AREA)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

array=[]
array.insert(0,img1)
arr=np.asarray(array)
arr = arr.reshape(1,48,48,1)
p=model.predict(arr)

mylist=np.array(p).tolist()
ind = np.argmax(mylist)

if ind==0: a='Angry'
elif ind==1: a='Dsigust'
elif ind==2: a='Fear'
elif ind==3: a='Happy'
elif ind==4: a='Sad'
elif ind==5: a='Surprise'
else : a='Neutral'

cv2.putText(img, a, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

plt.imshow(img)        #many attributes
plt.show()