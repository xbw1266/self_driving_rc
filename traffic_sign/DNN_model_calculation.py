#!/usr/bin/env python
# coding: utf-8

# This is my first attempt on this database, some parts of the kernel is based on Linus W. kernel, mostly how to access de data. It implements a very easy DNN achieving 96% of accuracy in the test set.
# I hope that this can be helpful as a starting point for anyone in the future. I will try to improve it in future revisions.

# In[345]:


# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os

INPUT_SIZE = 30
image_dir = '../../traffic_data'

# In[376]:


from scipy import ndimage
def augment_image(image):
    max_n = 15
    min_n = -15
    noise_add = np.random.uniform(min_n, max_n, image.shape)
    rotate_add = np.random.uniform(min_n, max_n)
    trans_x = np.random.uniform(10, 10)
    trans_y = np.random.uniform(10, 10)
#     print(rotate_add)

    #add noise to image
    noisy_img = np.add(image,noise_add)
    image = np.clip(noisy_img, 0, 255)
    image = np.int_(image)

    #rotation
    image = ndimage.rotate(image, rotate_add,reshape=False)
    #tanslation
    image = ndimage.shift(image,(trans_x,trans_y,0))

    return image


# In[379]:


#test

image = cv2.imread(image_dir+'/pi_image/2.png')
image = cv2.resize(image, (INPUT_SIZE,INPUT_SIZE))
print(image.shape)
print(type(image))
image = augment_image(image)
print(image.shape)
print(type(image))
plt.imshow(image)
plt.show()


# In[380]:


# Reading the input images and putting them into a numpy array
data=[]
labels=[]

channels = 3
classes = 43

for i in range(classes) :
    path = (image_dir+"/train/{0}/").format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((INPUT_SIZE, INPUT_SIZE))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

            
#augmentation
for i in range(classes) :
    path = "./train/{0}/".format(i)
    print("add augmented data",path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((INPUT_SIZE, INPUT_SIZE))
            size_image = augment_image(size_image)
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")          
            
Cells=np.array(data)
labels=np.array(labels)

#Randomize the order of the input images
s=np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]


# In[382]:


#Spliting the images into train and validation sets
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255


#Using one hote encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)


# In[383]:


#Definition of the DNN model

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)


# In[ ]:


#using ten epochs for the training and saving the accuracy for each epoch
epochs = 5

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))
model.save('german_model.h5')
#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()





