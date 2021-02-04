import numpy as np
from keras.layers import  Conv2D, Dense, Flatten, MaxPool2D, ZeroPadding2D, Conv3D
from keras.models import Sequential
from IPython import embed

# META-LAYER 1
model1 = Sequential()
model1.add(Conv3D(64, (5,5,5), strides=2, activation='relu',padding='same',input_shape=(10,8,10,5)))
model1.add(Conv3D(128, (3,3,3), strides=1, activation='relu',padding='same'))
model1.add(Conv3D(256, (3,3,3), strides=2, activation=None,padding='same'))

model2 = Sequential()
model2.add(Conv3D(256, (4,4,4), strides=3, activation=None,padding='valid',input_shape=(10,8,10,5)))

#embed()

# META-LAYER 2
model3 = Sequential()
model3.add(Conv3D(256, (2,1,2), strides=2, activation='relu',padding='same',input_shape=(3,2,3,256)))
model3.add(Conv3D(512, (2,1,2), strides=1, activation=None,padding='valid'))

model4 = Sequential()
model4.add(Conv3D(512, (3,2,3), strides=1, activation=None,padding='valid',input_shape=(3,2,3,256)))

embed()
