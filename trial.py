import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
import numpy as np

#training data
train=pd.read_csv('data/legend.csv')

x_train = train.iloc[:,1].values
y_train = train.iloc[:,2].values

y_train=LabelEncoder().fit_transform(y_train)
list1=[]

for i in range (0,13690):
    image_path=x_train[i]
    image=cv2.imread('/home/jak/Desktop/facial_expressions/images/'+image_path,0)
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    list1.append(image)


#testing data
test=pd.read_csv('data/500_picts_satz.csv')

x_test=test.iloc[:,1].values
y_test=test.iloc[:,2].values

y_test=LabelEncoder().fit_transform(y_test)

list2=[]
for i in range (0,499):
    image_path=x_test[i]
    image=cv2.imread('/home/jak/Desktop/facial_expressions/images/'+image_path,0)
    #image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    list2.append(image)
    

x1 = np.asarray(list1)
x2 = np.asarray(list2)

x1 = x1.reshape(13690,350,350,1)
x2 = x2.reshape(499,350,350,1)

from keras.models import Sequential
from keras.layers import Convolution2D    as Conv2D #for dealing with pictures
from keras.layers import MaxPooling2D       #pool maps reduces the sizeof map
from keras.layers import Flatten         #flatten the maps
from keras.layers import Dense         

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(350,350,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(14, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x1, y_train, validation_data=(x2, y_test), epochs=3)

