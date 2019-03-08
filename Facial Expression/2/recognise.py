import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#training data
train=pd.read_csv('fer2013.csv')

x_train = train.iloc[:28709,1].values
y_train = train.iloc[:28709,0:1].values

y_train=OneHotEncoder(categories='auto').fit_transform(y_train).toarray() 

x_test = train.iloc[28709:,1].values
y_test = train.iloc[28709:,0:1].values

y_test=OneHotEncoder(categories='auto').fit_transform(y_test).toarray() 


def arr_to_image(x):
    arr=x.split()   
    arr=np.array(arr)   
    arr=arr.astype(int)    
    a = np.reshape(arr, (48,48))  
    return a


for i in range(0,28709):
     x_train[i]=arr_to_image(x_train[i])
     
for i in range(0,7178):     
     x_test[i]=arr_to_image(x_test[i])

x1=list(x_train)
x2=list(x_test)
    
x1 = np.asarray(x1,dtype=np.uint8)
x2 = np.asarray(x2,dtype=np.uint8)

x1 = x1.reshape(28709,48,48,1)
x2 = x2.reshape(7178,48,48,1)

from keras.models import Sequential
from keras.layers import Convolution2D    as Conv2D #for dealing with pictures
from keras.layers import Flatten         #flatten the maps
from keras.layers import Dense         

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))


model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x1, y_train, validation_data=(x2, y_test), epochs=100)

"""
import pickle
with open('/home/jak/Desktop/face', 'wb') as f:
    pickle.dump(model,f)"""
