import tensorflow as tf
import numpy as np
import os
import json
import tensorflow_datasets as tfds
import tempfile
from tensorflow.experimental import dtensor
from typing import Tuple

# DTensor model creation
class Dense(tf.Module):
    def __init__(self, input_size, output_size, init_seed, weight_layout, activation=None):
        super().__init__()
        random_normal_initializer = tf.function(tf.random.stateless_normal)
        self.weight = dtensor.DVariable(
            dtensor.call_with_layout(
                random_normal_initializer, weight_layout,
                shape=[input_size, output_size],
                seed=init_seed
            ))
        if activation is None:
            activation = lambda x:x
        self.activation = activation
        bias_layout = weight_layout.delete([0])
        self.bias = dtensor.DVariable(
            dtensor.call_with_layout(tf.zeros, bias_layout, [output_size]))

    def __call__(self, x):
        y = tf.matmul(x, self.weight) + self.bias
        y = self.activation(y)
        return y

class BatchNorm(tf.Module):
    def __call__(self, x, training=True):
        mean, variance = tf.nn.moments(x, axes=[0])
        return tf.nn.batch_normalization(x, mean, variance, 0.0, 1.0, 1e-5)

class MLP(tf.Module):
    def __init__(self, dense_layouts: Tuple[dtensor.Layout, dtensor.Layout]):
        super().__init__()
        self.dense1 = Dense(1200, 48, (1, 2), dense_layouts[0], activation=tf.nn.relu)
        self.bn = BatchNorm()
        self.dense2 = Dense(48, 2, (3, 4), dense_layouts[1])

    def __call__(self, x):
        y = self.dense1(x)
        y = self.bn(y)
        y = self.dense2(y)
        return y

# Starts distributed connection
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ['10.0.2.15', '10.0.0.238']
    },
    'task': {'type': 'worker', 'index': 1}
})

# Initializes MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Setup DTensor training
configure_virtual_cpus(8)
DEVICES = [f'CPU:{i}' for i in range(8)]
WORLD = dtensor.create_mesh([("world", 8)], devices=DEVICES)

# Data Parellel Model Training
with strategy.scope():
    mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)
    model = MLP([
        dtensor.Layout.replicated(WORLD, rank=2),
        dtensor.Layout.replicated(WORLD, rank=2)
    ])

    train_data = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)
    text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200)
    text_vectorization.adapt(data=train_data.map(lambda x: x['text']))

    def vectorize(features):
        return text_vectorization(features['text']), features['label']

    train_data_vec = train_data.map(vectorize)

    @tf.function
    def train_step(model, x, y, learning_rate=tf.constant(1e-4)):
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = tf.reduce_sum(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits, labels=y))
        parameters = model.trainable_variables
        gradients = tape.gradient(loss, parameters)
        for parameter, parameter_gradient in zip(parameters, gradients):
            parameter.assign_sub(learning_rate * parameter_gradient)
        accuracy = 1.0 - tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)) / x.shape[0]
        loss_per_sample = loss / len(x)
        return {'loss': loss_per_sample, 'accuracy': accuracy}

    def repack_batch(x, y, mesh):
        x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
        y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
        return x, y

    CHECKPOINT_DIR = tempfile.mkdtemp()
    def start_checkpoint_manager(model):
        ckpt = tf.train.Checkpoint(root=model)
        manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)
        if manager.latest_checkpoint:
            print("Restoring a checkpoint")
            ckpt.restore(manager.latest_checkpoint).assert_consumed()
        else:
            print("New training")
        return manager

    num_epochs = 2
    manager = start_checkpoint_manager(model)

    for epoch in range(num_epochs):
        step = 0
        pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()), stateful_metrics=[])
        metrics = {'epoch': epoch}
        for x,y in train_data_vec:
            x, y = repack_batch(x, y, mesh)
            metrics.update(train_step(model, x, y, 1e-2))
            pbar.update(step, values=metrics.items(), finalize=False)
            step += 1
        manager.save()
        pbar.update(step, values=metrics.items(), finalize=True)