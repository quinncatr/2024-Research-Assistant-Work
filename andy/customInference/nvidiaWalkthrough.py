import tensorflow as tf
from tensorflow.keras.applications import ResNet50

tf.enable_eager_execution()

model_dir = './tmp_savedmodels/resnet50_saved_model'
model = ResNet50(include_top=True, weights='imagenet')

model.save(model_dir) 

import numpy as np

BATCH_SIZE = 32

dummy_input_batch = np.zeros((BATCH_SIZE, 224, 224, 3))

PRECISION = "FP32"

from helper import ModelOptimizer # using the helper from <URL>

model_dir = './tmp_savedmodels/resnet50_saved_model'

opt_model = ModelOptimizer(model_dir)

model_fp32 = opt_model.convert(model_dir+'_FP32', precision=PRECISION)

model_fp32.predict(dummy_input_batch)

model.predict(dummy_input_batch)
model_fp32.predict(dummy_input_batch)

#%%timeit

model.predict_on_batch(dummy_input_batch)

#%%timeit

model_fp32.predict(dummy_input_batch)
