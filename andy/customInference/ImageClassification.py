import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd
import vpi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.enable_eager_execution()

model = tf.keras.applications.MobileNetV2(weights='imagenet')
model.summary()

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path, top_k = 5):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top_k)[0]  # Decode the predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i + 1}: {label} ({score*100:.2f}%)")

def warmupLoop(processor):
    with processor:
        img1_path = 'Library.png'
        predict_image(img1_path, top_k = 7)

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
            img1_path = 'Library.png'
            predict_image(img1_path, top_k = 7)
            img2_path = 'cat.png'
            print('\n')
            predict_image(img2_path, top_k = 7)
            end = timer()

    elapsedTime = round((end - start), 5)
    current = energy['tot'][4]
    power = energy['tot'][2]
    voltage = energy["tot"][8]

    #print("Time to get inference results using "+ str(processor) + ": " + str(elapsedTime) + " seconds.")
    #print("Power consumed during inference" + str(processor) + ": " + str(power) + " milliwatts.")
    #print("Voltage drawn suring inference" + str(processor) + ": " + str(voltage) + " millivolts.\n")

print("GPU Warmup Loop:")
warmupLoop(vpi.Backend.CUDA)

print("CPU Warmup Loop:")
warmupLoop(vpi.Backend.CPU)

def graph():
    gpuTime = [None] * 10
    gpuPower = [None] * 10
    cpuTime = [None] * 10
    cpuPower = [None] * 10

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

graph()