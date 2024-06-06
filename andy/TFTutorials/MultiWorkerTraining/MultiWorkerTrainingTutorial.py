import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
os.environ.pop('TF_CONFIG', None)

if '.' not in sys.path:
    sys.path.insert(0, '.')

import tensorflow as tf

import mnist_setup

batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs = 3, steps_per_epoch = 70)