import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.enable_eager_execution()

#TODO: start recording time, energy etc on cpu and gpu
#TODO: Play with cpu v gpu distribution
#TODO: play with vic in the preprocessing stuff
#TODO Play around with different datasets and sizes
#TODO: plot data

#Can load data this way, tfds.load() will go find the data and download it for you
# try changing it to 'Cars196' to see if it works with other datasets
dataset_name = 'cifar10'
(train_dataset, test_dataset), dataset_info = tfds.load(name=dataset_name,
                                                        split=['train', 'test'],
                                                        shuffle_files=True,
                                                        with_info=True,
                                                        as_supervised=True)

image, label = next(iter(train_dataset.take(1)))

plt.imshow(image)
plt.title(label.numpy())
plt.axis('off')
plt.savefig(str('Graph').split()[0]+'.png')

num_classes = dataset_info.features['label'].num_classes
num_classes

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess_data)
test_dataset = test_dataset.map(preprocess_data)

input_dim = (32, 32, 3)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_dim),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.summary()

tf.keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

batch_size = 128
num_epochs = 10

train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

loss, accuracy = model.evaluate(test_dataset)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

new_image = tf.constant(np.random.rand(32, 32, 3), dtype=tf.float64)
new_image = tf.expand_dims(new_image, axis=0)

predictions = model.predict(new_image)
pred_label = tf.argmax(predictions, axis =1)
pred_label.numpy()
