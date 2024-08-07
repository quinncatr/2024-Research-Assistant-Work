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
    global Current
    global Power
    global Voltage
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
    Current = energy['tot'][4]
    Power = energy['tot'][2]
    Voltage = energy["tot"][8]

    #print("Time to get inference results using "+ str(processor) + ": " + str(elapsedTime) + " seconds.")
    #print("Power consumed during inference" + str(processor) + ": " + str(Power) + " milliwatts.")
    #print("Voltage drawn suring inference" + str(processor) + ": " + str(Voltage) + " millivolts.\n")

print("GPU Warmup Loop:")
warmupLoop(vpi.Backend.CUDA)
#inference(vpi.Backend.CUDA)

print("CPU Warmup Loop:")
warmupLoop(vpi.Backend.CPU)
#inference(vpi.Backend.CPU)


def graph():
    x1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x2 = [None] * 10
    y2 = [None] * 10

    for t in range(10):
        inference(vpi.Backend.CUDA)
        #GPU Time for 10 seperate runs
        x2[t] = elapsedTime
        inference(vpi.Backend.CPU)
        #CPU Time for 10 seperate runs
        y2[t] = elapsedTime

    n=10
    r = np.arange(n) 
    width = 0.25

    plt.bar(r, x2, color = 'g', 
            width = width, edgecolor = 'black', 
            label='GPU Execution Time') 
    plt.bar(r + width, y2, color = 'b', 
            width = width, edgecolor = 'black', 
            label='CPU Execution Time') 

    plt.xticks(r + width/2,['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) 

    plt.xlabel('Run Number')
    plt.ylabel('Elapsed Time (s)')
    plt.title('CPU & GPU Inference Execution Time per Run')
    plt.legend()
    plt.savefig(str('CPU-GPU-Execution-Time-Run').split()[0]+'.png')
