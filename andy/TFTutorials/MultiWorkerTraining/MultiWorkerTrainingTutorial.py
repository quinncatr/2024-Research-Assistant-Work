import json
import os
import sys

#os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
#os.environ.pop('TF_CONFIG', None)

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


strategy = tf.distribute.MultiWorkerMirrorStrategy()
with strategy.scope():
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

