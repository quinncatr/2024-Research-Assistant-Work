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
# with tf.device('/GPU:0'):
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
# with tf.device('/CPU:0'):
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

def gpu_model_fit():
    global startGPU
    global endGPU
    global gpuData
    global gpuPower
    global gpuEnergy
    with jtop() as jetson:
            gpuData = pd.DataFrame(jetson.stats, index = [0])
            power = pd.DataFrame(jetson.power)
            gpuPower = power['tot'][2]
    with tf.device('/GPU:0'):
        startGPU = timer()
        model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
        endGPU = timer()
    gpuEnergy = round(gpuPower * (endGPU - startGPU), 5)

def cpu_model_fit():
    global startCPU
    global endCPU
    global cpuData
    global cpuPower
    global cpuEnergy
    with jtop() as jetson:
            cpuData = pd.DataFrame(jetson.stats, index = [0])
            power = pd.DataFrame(jetson.power)
            cpuPower = power['tot'][2]
    with tf.device('/CPU:0'):
        startCPU = timer()
        model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
        endCPU = timer()
    cpuEnergy = round(cpuPower * (endCPU - startCPU), 5)

def distributed_model_fit():
    global startDist
    global endDist
    global distData
    global distPower
    global distEnergy
    with jtop() as jetson:
            distData = pd.DataFrame(jetson.stats, index = [0])
            power = pd.DataFrame(jetson.power)
            distPower = power['tot'][2]
    # with tf.device('/CPU:0') and tf.device('/GPU:0'):
    # startDist = timer()
    model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
    # endDist = timer()
    distEnergy = distPower * (endDist - startDist)

gpu_model_fit()
#cpu_model_fit()
#distributed_model_fit()

gpuTime = round(endGPU - startGPU, 5) 
gpuEnergy = round(gpuEnergy / 1000, 5)

#cpuTime = round(endCPU - startCPU, 5)
#cpuEnergy = round(cpuEnergy / 1000, 5)


print("\n-----Time to Completion-----\n")
print("Time to complete using the GPU: " + str(gpuTime) + " seconds.")
#print("Time to complete using the CPU: " + str(endCPU - startCPU) + " seconds")
#print("Time to complete when distributed between the CPU and GPU: " + str(endDist - startDist) + " seconds")

print("\n-----Power Consumption-----\n")
print("Power consumption using the GPU: " + str(gpuPower / 1000) + " watts")
#print("Power consumption using the CPU: " + str(cpuPower / 1000) + " watts")
#print("Power consumption distributed between the CPU and GPU: " + str(distPower) + " milliwatts")

print("\n-----Energy Usage-----\n")
print("Energy usage using the GPU: " + str(gpuEnergy) + " joules")
#print("Energy usage using the CPU: " + str(cpuEnergy / 1000) + " joules")
#print("Energy usage distributed between the CPU and GPU: " + str(distEnergy) + " millijoules")

print("\n-----Comparisons-----\n")


print("Time to create model: " + str(createTime) + " seconds.")

#TODO: distribute model between cpu and gpu (Like line 32-40)
#TODO: distribute layers between cpu and gpu (Like line 12)
    # do data vs model parallelization? I think previous two todos are both model
#TODO: find out if vic could help with this stuff
#TODO: Distribute 1 layer between cpu and gpu
#TODO: fix power printing not matching UI
