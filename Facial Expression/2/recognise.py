import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#training data
train=pd.read_csv('fer2013.csv')

x_train = train.iloc[:28709,1].values
y_train = train.iloc[:28709,0:1].values

#y_train=OneHotEncoder(categories='auto').fit_transform(y_train).toarray() 

x_test = train.iloc[28709:,1].values
y_test = train.iloc[28709:,0:1].values

#y_test=OneHotEncoder(categories='auto').fit_transform(y_test).toarray() 


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
from keras.layers import Flatten   , MaxPooling2D      #flatten the maps
from keras.layers import Dense         
"""
model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))


model.add(Flatten())
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x1, y_train, validation_data=(x2, y_test), epochs=100)

import pickle
with open('/home/jak/Desktop/face', 'wb') as f:
    pickle.dump(model,f)
"""


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(48,48,1)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='softmax'))

model.add(Dense(7 , activation='softmax'))

#model.compile(loss='binary_crossentropy',
#              optimizer='adam' ,
#              metrics=['accuracy'])

print(model.summary())

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#model.fit(x1, y_train, validation_data=(x2, y_test), epochs=100)

batch_size = 128
epochs = 14

model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD' , metrics=['accuracy'])
steps_per_epoch = len(x1) / batch_size
validation_steps = len(y_test)/ batch_size

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.0,  
        width_shift_range=0.1,  
        height_shift_range=0.1,  
        horizontal_flip=False, 
        vertical_flip=False)  

datagen.fit(x1)

history = model.fit_generator(datagen.flow(x1, y_train, batch_size=batch_size),
                    steps_per_epoch=x1.shape[0] / batch_size,
                    validation_data=(x2, y_test),
                    epochs = epochs, verbose = 2)

import pickle
with open('/home/jak/Desktop/face', 'w') as f:
    pickle.dump(model,f)