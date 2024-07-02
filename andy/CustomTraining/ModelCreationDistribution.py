import tensorflow as tf
import tensorflow_datasets as tfds
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd

strategy = tf.distribute.MirroredStrategy(devices = ['/GPU:0', '/CPU:0'])
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

inputs = tf.keras.Input(shape = (28, 28, 1))
conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)

def cpu_layers():
    with strategy.scope():
        with tf.device('/cpu:0'):
            global pool2
            pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
            conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
            pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

def gpu_layers():
    with strategy.scope():
        with tf.device('/gpu:0'):
            global output
            flatten = tf.keras.layers.Flatten()(pool2)
            #splitFlatten = tf.split(flatten, 2)
            #x = tf.concat([flatten, splitFlatten], axis=2)
            dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
            output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)


def create_model():
    global createTime
    #within scope:
    with strategy.scope():
        createStart = timer()
        cpu_layers()
        gpu_layers()
        model = tf.keras.Model(inputs = inputs, outputs = output)
        createEnd = timer()
    createTime = round((createEnd - createStart), 5)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    print("Time to create model within scope: " + str(createTime) + " seconds")

    #outside of scope:
    createStart = timer()
    cpu_layers()
    gpu_layers()
    createEnd = timer()
    createTime = round((createEnd - createStart), 5)
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    print("Time to create model outside of scope: " + str(createTime) + " seconds")
    
    return model


model1 = create_model()

#TODO: Find out why changing layers from cpu to gpu methods and vice versa doesn't change results
#TODO: distribute layers between cpu and gpu (Like model.create)
# tf.split and then concatenate or maybe need to create custom layer? 
#TODO: find out if vic could help with this stuff
#TODO: Distribute 1 layer between cpu and gpu
#TODO: Make a ml model that will determine optimal cpu, gpu, vic ratios
# Create a reinforcement learning algorithm to dynamically control how TensorFlow distributes between CPU and GPU to optimize resource usage
# - The program the ml distributes doesn;t have to be ml, it could just be some operations distributed in various ways
