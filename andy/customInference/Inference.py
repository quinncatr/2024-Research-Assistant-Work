import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd

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

#Sequential inference(default processors)
start = timer()
with jtop() as jetson:
    data = pd.DataFrame(jetson.stats, index = [0])
    energy = pd.DataFrame(jetson.power)
    print("Image 1: \n")
    img1_path = 'Library.png'
    predict_image(img1_path, top_k = 7)
    print("\nImage 2: \n")
    img2_path = 'cat.png'
    predict_image(img2_path, top_k = 7)
    end = timer()

elapsedTime = end - start
current = energy['tot'][4]
power = energy['tot'][2]
voltage = energy["tot"][8]


print("Time to get inference results: " + str(elapsedTime) + " seconds.")
print("Power consumed during inference: " + str(power) + " milliwatts.")
print("Voltage drawn suring inference: " + str(voltage) + " millivolts.")


from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os

# Load a pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

model_name = "model"
baseModel = os.path.join(os.path.dirname(os.path.abspath(__file__)),(model_name))
pb_model  = os.path.join(os.path.dirname(os.path.abspath(__file__)),(model_name+"_pb")) 
trt_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),(model_name+"_trt")) 

if not os.path.exists(pb_model):
    os.mkdir(pb_model)

if not os.path.exists(trt_model):
    os.mkdir(trt_model)

if not os.path.exists(baseModel):
    os.mkdir(baseModel)

model.save("test_saved_model.keras")