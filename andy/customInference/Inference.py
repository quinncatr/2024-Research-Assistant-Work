import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd
import vpi

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

def inference(processor):
    global elapsedTime
    global Current
    global Power
    global Voltage
    with processor:
        with jtop() as jetson:
            data = pd.DataFrame(jetson.stats, index = [0])
            energy = pd.DataFrame(jetson.power)
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

    print("Time to get inference results using "+ str(processor) + ": " + str(elapsedTime) + " seconds.")
    print("Power consumed during inference" + str(processor) + ": " + str(Power) + " milliwatts.")
    print("Voltage drawn suring inference" + str(processor) + ": " + str(Voltage) + " millivolts.\n")

#warmup loop

print("CPU Warmup Loop:")
inference(vpi.Backend.CPU)

print("GPU Warmup Loop:")
inference(vpi.Backend.CUDA)

inference(vpi.Backend.CPU)
inference(vpi.Backend.CUDA)


