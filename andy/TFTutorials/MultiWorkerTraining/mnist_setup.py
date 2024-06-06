#%%writefile mnist_setup.py

import os
import tensorflow as tf
import numpy as np
import sys
import json

def mnist_dataset(batch_size):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape = (28, 28)),
        tf.keras.layers.Reshape(target_shape = (28, 28, 1)),
        tf.keras.layers.Conv2D(32, 2, activation = 'relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001),
        metrics = ['accuracy'])
    return model

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ.pop('TF_CONFIG', None)

if '.' not in sys.path:
    sys.path.insert(0, '.')

import tensorflow as tf

import mnist_setup

#This single worker test makes sure that everything works before going onto multi worker
#NOTE: this single worker makes it run on the CPU, not the GPU due to line 5 
batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs = 3, steps_per_epoch = 70)


tf_config = {
    'cluster': {
        'worker':['localhost:1', 'localhost:2']
    },
    'task': {'type': 'worker', 'index': 0}
}

json.dumps(tf_config)

tf_config['task']['index'] = 1


strategy = tf.distribute.MultiWorkerMirroredStrategy()
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

