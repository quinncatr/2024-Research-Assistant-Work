import tensorflow as tf
import tensorflow_datasets as tfds
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd

def create_model():
    global createTime
    inputs = tf.keras.Input(shape = (28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)
    createStart = timer()
    with tf.device('/CPU:0'):
        pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    with tf.device('/GPU:0'):    
        flatten = tf.keras.layers.Flatten()(pool2)
        dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
        output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    createEnd = timer()
    createTime = round((createEnd - createStart), 5)
    return model

model = create_model()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


print(createTime)

#TODO: distribute layers between cpu and gpu (Like model.create)
#TODO: find out if vic could help with this stuff
#TODO: Distribute 1 layer between cpu and gpu
