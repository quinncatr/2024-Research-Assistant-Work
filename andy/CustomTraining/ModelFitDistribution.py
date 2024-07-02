import tensorflow as tf
import tensorflow_datasets as tfds
from timeit import default_timer as timer
from jtop import jtop
import pandas as pd
import numpy as np

def create_model():
    global createTime
    inputs = tf.keras.Input(shape = (28, 28, 1))
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu')(inputs)
    createStart = timer()
    with tf.device('/GPU:0'):
        pool1 = tf.keras.layers.MaxPooling2D(2, 2)(conv1)
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        flatten = tf.keras.layers.Flatten()(pool2)
        dense1 = tf.keras.layers.Dense((64), activation = 'relu')(flatten)
        output = tf.keras.layers.Dense(10, activation = 'softmax')(dense1)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    createEnd = timer()
    createTime = round((createEnd - createStart), 5)
    return model

model = create_model()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

#This method is distributed biscally at random right now for testing purposes
#Different combonations of where "tf.device" is produces significantly different results which is a good sign
def custom_fit(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    with tf.device('/gpu:0'):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_losses = []
            train_accuracies = []

            for batch_start in range(0, len(x_train), batch_size):
                batch_end = min(batch_start + batch_size, len(x_train))
                x_batch = x_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                loss, accuracy = model.train_on_batch(x_batch, y_batch)

                with tf.device('/cpu:0'):
                    train_losses.append(loss)
                    train_accuracies.append(accuracy)

            with tf.device('/cpu:0'):
                avg_train_loss = np.mean(train_losses)
                avg_train_accuracy = np.mean(train_accuracies)
                print(f" Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")

                val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose = 0)
                print(f" Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return model

def model_fit_single_processor(accelerator):
    global start
    global end
    global time
    global data
    global power
    global energy
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    with jtop() as jetson:
            data = pd.DataFrame(jetson.stats, index = [0])
            powerInfo = pd.DataFrame(jetson.power)
            power = powerInfo['tot'][2]
    with tf.device(accelerator):
        start = timer()
        model.fit(x_train, y_train, epochs = 5, batch_size = 64, validation_data = (x_test, y_test))
        end = timer()
    time = round(end - start, 5)
    energy = round(power * (end - start), 5)

def distributed_model_fit(accelerator):
    global start
    global end
    global time
    global data
    global power
    global energy
    with jtop() as jetson:
            data = pd.DataFrame(jetson.stats, index = [0])
            powerInfo = pd.DataFrame(jetson.power)
            power = powerInfo['tot'][2]
    with tf.device(accelerator):
        start = timer()
        custom_fit(model, x_train, y_train, x_test, y_test, 5, 64)
        end = timer()
    time = round(end - start, 5)
    energy = round(power * (end - start), 5)

gpu = '/GPU:0'
cpu = '/CPU:0'

#warm up loop
distributed_model_fit(gpu)
distributed_model_fit(cpu)

model_fit_single_processor(gpu)
gpuTime = round(end - start, 5) 
gpuEnergy = round(energy / 1000, 5)
gpuPower = power

model_fit_single_processor(cpu)
cpuTime = round(end - start, 5)
cpuEnergy = round(energy / 1000, 5)
cpuPower = power

distributed_model_fit(gpu)
distGpuTime = round(end - start, 5)
distGpuEnergy = round(energy / 1000, 5)
distGpuPower = power

distributed_model_fit(cpu)
distCpuTime = round(end - start, 5)
distCpuEnergy = round(energy / 1000, 5)
distCpuPower = power

print("\n-----Time to Completion-----\n")
print("Time to complete using the GPU: " + str(gpuTime) + " seconds.")
print("Time to complete using custom fit on the GPU: " + str(distGpuTime) + " seconds.")
print("Time to complete using the CPU: " + str(cpuTime) + " seconds.")
print("Time to complete using custom fit on the CPU: " + str(distCpuTime) + " seconds.")

print("\n-----Power Consumption-----\n")
print("Power consumption using the GPU: " + str(gpuPower / 1000) + " watts.")
print("Power consumption using custom fit on the GPU: " + str(distGpuPower / 1000) + " watts.")
print("Power consumption using the CPU: " + str(cpuPower / 1000) + " watts.")
print("Power consumption using custom fit on the CPU: " + str(distCpuPower / 1000) + " watts.")

print("\n-----Energy Usage-----\n")
print("Energy usage using the GPU: " + str(gpuEnergy) + " joules.")
print("Energy usage using custom fit on the GPU: " + str(distGpuEnergy) + " joules.")
print("Energy usage using the CPU: " + str(cpuEnergy) + " joules.")
print("Energy usage using custom fit on the CPU: " + str(distCpuEnergy) + " joules.")


cpuGpuTimeDiff = abs(cpuTime - gpuTime)
customCpuGpuTimeDiff = abs(distCpuTime - distGpuTime)
cpuTimeDiff = round(abs(cpuTime - distCpuTime), 2)
gpuTimeDiff = round(abs(gpuTime - distGpuTime), 2)

cpuGpuPowerDiff = abs(cpuPower - gpuPower)
customCpuGpuPowerDiff = abs(distGpuPower - distCpuPower)
cpuPowerDiff = abs(cpuPower - distCpuPower)
gpuPowerDiff = abs(gpuPower - distGpuPower)

cpuGpuPercentTimeDiff = round(((cpuGpuTimeDiff / ((cpuTime + gpuTime) / 2)) * 100), 2)
customCpuGpuPercentTimeDiff = round(((customCpuGpuTimeDiff / ((cpuTime + gpuTime) / 2)) * 100), 2)

cpuGpuPercentPowerDiff = round(((cpuGpuPowerDiff / ((cpuPower + gpuPower) / 2)) * 100), 2)
customCpuGpuPercentPowerDiff = round(((customCpuGpuPowerDiff / ((cpuPower + gpuPower) / 2)) * 100), 2)

cpuPercentTimeDiff = round(((cpuTimeDiff / ((cpuTime + distCpuTime) / 2)) * 100), 2)
cpuPercentPowerDiff = round(((cpuPowerDiff / ((cpuPower + distCpuPower) / 2)) * 100), 2)

gpuPercentTimeDiff = round(((gpuTimeDiff / ((gpuTime + distGpuTime) / 2)) * 100), 2)
gpuPercentPowerDiff = round(((gpuPowerDiff / ((gpuPower + distGpuPower) / 2)) * 100), 2)

print("\n-----Comparisons-----\n")
print("The custom_fit(model) method running on the CPU was " + str(cpuPercentTimeDiff) + "% slower than the model.fit() method on the CPU.")
print("The custom_fit(model) method running on the GPU was " + str(gpuPercentTimeDiff) + "% slower than the model.fit() method on the GPU.")

print("The custom_fit(model) method running on the GPU was " + str(customCpuGpuPercentTimeDiff) + "% faster than the custom_fit(model) method on the CPU.")
print("The model.fit() method running on the GPU was " + str(cpuGpuPercentTimeDiff) + "% slower than the model.fit() method on the CPU.")

print("The custom_fit(model) method running on the CPU consumed " + str(cpuPercentPowerDiff) + "% more power than the model.fit() method on the CPU.")
print("The custom_fit(model) method running on the GPU consumed " + str(gpuPercentPowerDiff) + "% more power than the model.fit() method on the GPU.")

print("The custom_fit(model) method running on the CPU consumed " + str(customCpuGpuPercentPowerDiff) + "% less power than the custom_fit(model) method on the GPU.")
print("The model.fit() method running on the CPU consumed " + str(cpuGpuPercentPowerDiff) + "% more power than the model.fit() method on the GPU.")

#TODO: Distribute between vic cpu/gpu
#TODO: Find out why VIC and CPU stats are basically identical
# - Program doesn't identify VIC as a processor when you list physoical devices, so it falls back to CPU