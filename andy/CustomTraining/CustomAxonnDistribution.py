import tensorflow as tf
from timeit import default_timer as timer

device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device[0], True)
tf.config.experimental.set_virtual_device_configuration(device[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def create_model():
    inputs = tf.keras.Input(shape = (28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)

    with tf.device('/GPU:0'):
        pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(pool2)
    
    dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
    output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    return model

model = create_model()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

startFit = timer()
model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
endFit = timer()

print("TIME to FIT: " + str(endFit - startFit))

