import tensorflow as tf
import tensorflow_datasets as tfds
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def create_model():
    inputs = tf.keras.Input(shape = (28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)

    #with tf.device('/GPU:0'):
    pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    flatten = tf.keras.layers.Flatten()(pool2)
    
    dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)

    model = tf.keras.Model(inputs = inputs, outputs = output)
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
    with jtop() as jetson:
            gpuData = pd.DataFrame(jetson.stats, index = [0])
            gpuEnergy = pd.DataFrame(jetson.power)
            gpuPower = gpuEnergy['tot'][7]
    with tf.device('/GPU:0'):
        startGPU = timer()
        model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
        endGPU = timer()

def cpu_model_fit():
    global startCPU
    global endCPU
    global cpuData
    global cpuPower
    with jtop() as jetson:
            cpuData = pd.DataFrame(jetson.stats, index = [0])
            cpuEnergy = pd.DataFrame(jetson.power)
            cpuPower = cpuEnergy['tot'][7]
    with tf.device('/CPU:0'):
        startCPU = timer()
        model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
        endCPU = timer()

def distributed_model_fit():
    return

gpu_model_fit()
cpu_model_fit()
distributed_model_fit()

print("\n-----Time to Completion-----\n")
print("Time to complete using the GPU: " + str(endGPU - startGPU) + " seconds.")
print("Time to complete using the CPU: " + str(endCPU - startCPU) + " seconds")
print("Time to complete when distributed between the CPU and GPU: " + str(-1) + " seconds")

print("\n-----Power Consumption-----\n")
print("Power consumption using the GPU: " + str(gpuPower) + " milliwatts")
print("Power consumption using the CPU: " + str(cpuPower) + " milliwatts")
print("Power consumption distributed between the CPU and GPU: " + str(-1) + " milliwatts")

print("\n-----Comparisons-----\n")

#TODO: add power consumption numbers
#TODO: distribute model between cpu and gpu (Like line 32-40)
#TODO: distribute layers between cpu and gpu (Like line 12)
    # do data vs model parallelization? I think previous two todos are both model
#TODO: Figour out how to run it on a other datasets
#TODO: Distribute 1 layer between cpu and gpu
