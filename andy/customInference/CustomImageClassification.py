import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jtop import jtop
import pandas as pd
import vpi
from timeit import default_timer as timer

tf.enable_eager_execution()

#TODO: Do axonn layer splitting for the inference

#TODO: Play with cpu v gpu distribution
#TODO: play with vic in the preprocessing stuff
#TODO Play around with different datasets and sizes

#Can load data this way, tfds.load() will go find the data and download it for you
# try changing it to 'Cars196' to see if it works with other datasets
dataset_name = 'cifar10'
(train_dataset, test_dataset), dataset_info = tfds.load(name=dataset_name,
                                                        split=['train', 'test'],
                                                        shuffle_files=True,
                                                        with_info=True,
                                                        as_supervised=True)

image, label = next(iter(train_dataset.take(1)))
'''
plt.imshow(image)
plt.title(label.numpy())
plt.axis('off')
plt.savefig(str('Graph').split()[0]+'.png')
'''

num_classes = dataset_info.features['label'].num_classes
num_classes

def preprocess_data(image, label):
    with vpi.Backend.VIC:
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

train_dataset = train_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

input_dim = (32, 32, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

batch_size = 128
num_epochs = 10

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

loss, accuracy = model.evaluate(test_dataset)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

new_image = tf.constant(np.random.rand(32, 32, 3), dtype=tf.float64)
new_image = tf.expand_dims(new_image, axis=0)

def warmupLoop(processor):
    with processor:
        predictions = model.predict(new_image)
        pred_label = tf.argmax(predictions, axis =1)
        pred_label.numpy()

def inference(processor):
    global elapsedTime
    global current
    global power
    global voltage
    with jtop() as jetson:
        data = pd.DataFrame(jetson.stats, index = [0])
        energy = pd.DataFrame(jetson.power)
        with processor:
            start = timer()
            predictions = model.predict(new_image)
            pred_label = tf.argmax(predictions, axis =1)
            pred_label.numpy()
            end = timer()
    elapsedTime = round((end - start), 5)
    current = energy['tot'][4]
    power = energy['tot'][2]
    voltage = energy["tot"][8]

def graph():
    gpuTime = [None] * 10
    gpuPower = [None] * 10
    cpuTime = [None] * 10
    cpuPower = [None] * 10

    warmupLoop(vpi.Backend.CUDA)
    warmupLoop(vpi.Backend.CPU)

    for t in range(10):
        inference(vpi.Backend.CUDA)
        #GPU Time for 10 seperate runs
        gpuTime[t] = elapsedTime
        gpuPower[t] = power
        inference(vpi.Backend.CPU)
        #CPU Time for 10 seperate runs
        cpuTime[t] = elapsedTime
        cpuPower[t] = power

    n=10
    r = np.arange(n) 
    width = 0.25
    plt.xlabel('Run Number')

    gpuTimeBar = plt.bar(r, gpuTime, color = 'g', 
            width = width, edgecolor = 'black', 
            label='GPU Execution Time') 
    cpuTimeBar = plt.bar(r + width, cpuTime, color = 'b', 
            width = width, edgecolor = 'black', 
            label='CPU Execution Time') 

    plt.xticks(r + width/2,['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 

    timeY = plt.ylabel('Elapsed Time (s)')
    timeTitle = plt.title('CPU & GPU Inference Execution Time per Run')
    timeLegend = plt.legend(handles=[gpuTimeBar, cpuTimeBar])
    timeSave = plt.savefig(str('CPU-GPU-Execution-Time-Run').split()[0]+'.png')

    gpuTimeBar
    cpuTimeBar
    timeY
    timeTitle
    timeLegend
    timeSave

    gpuPowerBar = plt.bar(r, gpuPower, color = 'g', 
            width = width, edgecolor = 'black', 
            label='GPU Power Consumption') 
    cpuPowerBar = plt.bar(r + width, cpuPower, color = 'b', 
            width = width, edgecolor = 'black', 
            label='CPU Power Consumption') 

    plt.xticks(r + width/2,['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 

    powerY = plt.ylabel('Power Consumption (mw)')
    powerTitle = plt.title('CPU & GPU Inference Power Consumption per Run')
    powerLegend = plt.legend(handles=[gpuPowerBar, cpuPowerBar])
    powerSave = plt.savefig(str('CPU-GPU-Power-Consumption-Run').split()[0]+'.png')

    gpuPowerBar
    cpuPowerBar
    powerY
    powerTitle
    powerLegend
    powerSave

    averageCPUTime = sum(cpuTime) / len(cpuTime)
    averageCPUPower = sum(cpuPower) / len(cpuPower)

    averageGPUTime = sum(gpuTime) / len(gpuTime)
    averageGPUPower = sum(gpuPower) / len(gpuPower)

    print("Average Execution Time for CPU: " + str(averageCPUTime) + " seconds.")
    print("Average Power Consumption for CPU: " + str(averageCPUPower) + " milliwatts.")
    print("Average Execution Time for GPU: " + str(averageGPUTime) + " seconds.")
    print("Average Power Consumption for GPU: " + str(averageGPUPower) + " milliwatts.")

graph()
