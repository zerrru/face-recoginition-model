# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:49:25 2021

@author: user
"""
import keras
from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
#from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception as xc
from keras_applications.xception import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
#spacifying the path of images
train_path='Datasets/train'
test_path='Datasets/test'

IMAGW_SIZE=[224,224]
#creating the obgetct of resnet50 class
xcep=xc(input_shape=IMAGW_SIZE+[3],weights='imagenet',include_top=False)
for layers in xcep.layers:
    layers.trainable=False
    
    
    
folder=glob('Datasets/train/*')   
x=Flatten()(xcep.output)
prediction=Dense(2,activation='softmax')(x)
model=Model(inputs=xcep.input, outputs=prediction)


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
train_datgen=ImageDataGenerator(rescale=1./255,zoom_range=0.2,shear_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)



trainset=train_datgen.flow_from_directory('Datasets/train',
                                          target_size=(224,224),
                                          batch_size=32,
                                          class_mode='categorical')
testset=test_datagen.flow_from_directory('Datasets/test',
                                         target_size=(224,224),
                                         batch_size=32,
                                         class_mode='categorical')


t=model.fit_generator(trainset,
                      validation_data=testset,
                      epochs=2,
                      steps_per_epoch=len(trainset),
                      validation_steps=len(testset)
                      )






import tensorflow as tf
from keras.models import load_model


model.save('face_detection_model.h5')





