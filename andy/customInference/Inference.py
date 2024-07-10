import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load a pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
model.summary()

# Define a function to preprocess an image for model prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to predict the top 5 labels for an image
def predict_image(img_path, top_k = 5):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top_k)[0]  # Decode the predictions
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        print(f"{i + 1}: {label} ({score*100:.2f}%)")

# Example usage: Predict labels for an image
img_path = 'Library.png'
predict_image(img_path, top_k = 10)