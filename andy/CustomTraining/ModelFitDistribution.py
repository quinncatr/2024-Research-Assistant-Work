import tensorflow as tf
import tensorflow_datasets as tfds
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd
import vpi

def create_model():
    global createTime
    inputs = tf.keras.Input(shape = (28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)
    createStart = timer()
    with tf.device('/GPU:0'):
        pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(pool2)
        dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
        output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    createEnd = timer()
    createTime = round((createEnd - createStart), 5)
    return model

model = create_model()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

def generic_model_fit(accelerator):
    global start
    global end
    global time
    global data
    global power
    global energy
    with jtop() as jetson:
            data = pd.DataFrame(jetson.stats, index = [0])
            powerInfo = pd.DataFrame(jetson.power)
            power = powerInfo['tot'][2]
    if accelerator == '/GPU:0' or accelerator == '/CPU:0':
        with tf.device(accelerator):
            start = timer()
            model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
            end = timer()
    elif accelerator == vpi.Backend.VIC:
        with vpi.Backend.VIC:
            start = timer()
            model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
            end = timer()
    else:
        return
    time = round(end - start, 5)
    energy = round(power * (end - start), 5)

gpu = '/GPU:0'
cpu = '/CPU:0'
vic = 'vpi.Backend.VIC'

generic_model_fit(gpu)
gpuTime = round(end - start, 5) 
gpuEnergy = round(energy / 1000, 5)
gpuPower = power

generic_model_fit(cpu)
cpuTime = round(end - start, 5)
cpuEnergy = round(energy / 1000, 5)
cpuPower = power

generic_model_fit(vic)
vicTime = round(end - start, 5)
vicEnergy = round(energy / 1000, 5)
vicPower = power

print("\n-----Time to Completion-----\n")
print("Time to complete using the GPU: " + str(gpuTime) + " seconds.")
print("Time to complete using the CPU: " + str(cpuTime) + " seconds")
print("Time to complete using VIC: " + str(vicTime) + " seconds")

print("\n-----Power Consumption-----\n")
print("Power consumption using the GPU: " + str(gpuPower / 1000) + " watts")
print("Power consumption using the CPU: " + str(cpuPower / 1000) + " watts")
print("Power consumption using VIC: " + str(vicPower / 1000) + " watts")

print("\n-----Energy Usage-----\n")
print("Energy usage using the GPU: " + str(gpuEnergy) + " joules")
print("Energy usage using the CPU: " + str(cpuEnergy) + " joules")
print("Energy usage using VIC: " + str(vicEnergy) + " joules")

print("\n-----Comparisons-----\n")

#print("Time to create model on CPU: " + str(createTime) + " seconds.")
#print("Time to create model on GPU: " + str(createTime) + " seconds.")

#TODO: Distribute between vic cpu/gpu
#TODO: Distribute 1 layer between cpu and gpu
#TODO: Find out why VIC and CPU stats are basically identical