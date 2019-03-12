import numpy as np
import cv2
import pickle

with open('/home/jak/Desktop/face2', 'rb') as f:
    model = pickle.load(f)


#web cam
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
img1=frame
#img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

"""
imgpath="/home/jak/Desktop/1.jpg"
img1=cv2.imread(imgpath,0)
plt.imshow(img1,cmap="gray")        #many attributes
plt.show()"""



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
x=np.asarray(array)
x = x.reshape(1,48,48,1)
p=model.predict(x)


mylist=np.array(p).tolist()
ind = np.argmax(mylist)
print(ind+1)


if ind==0: print('Angry')
elif ind==1: print('Dsigust')
elif ind==2: print('Fear')
elif ind==3: print('Happy')
elif ind==4: print('Sad')
elif ind==5: print('Surprise')
else :print('Neutral')

