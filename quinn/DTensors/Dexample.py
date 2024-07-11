import tempfile
import numpy as np
import tensorflow_datasets as tfds

import tensorflow as tf

from tensorflow.experimental import dtensor

print('TensorFlow version:', tf.__version__)

def configure_virtual_cpus(ncpu):
  phy_devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(phy_devices[0], [
        tf.config.LogicalDeviceConfiguration(),
    ] * ncpu)

configure_virtual_cpus(8)
DEVICES = [f'CPU:{i}' for i in range(8)]

tf.config.list_logical_devices('CPU')