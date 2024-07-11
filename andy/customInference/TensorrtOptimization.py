import tensorflow as tf
from tensorflow import keras
import os

#from tensorflow.contrib.tensorrt import trt_convert
#import tensorflow.contrib.tensorrt as trt
#from tensorflow.python.compiler.tensorrt import trt
#import tensorrt as trt
#from tensorrt import trt_convert
#from tensorrt import trt_convert as trt

tf.enable_eager_execution()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), metrics = ['accuracy'])

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test, = x_train / 255.0, x_test / 255.0
x_train = tf.cast(x_train, dtype = tf.float32)
y_train = tf.cast(y_train, dtype = tf.float32)
x_test = tf.cast(x_test, dtype = tf.float32)
y_test = tf.cast(y_test, dtype = tf.float32)

model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test, verbose = 2)


saved_model_path = './savedModel/saved_model.pb'
try:
    model = tf.keras.models.load_model(saved_model_path)
    print("Model loaded successfully!")
except OSError as e:
    print(f"Error loading the model: {e}")


model_name = "simpleModel"
pb_model  = os.path.join(os.path.dirname(os.path.abspath(__file__)),(model_name+"_pb")) 
trt_model = os.path.join(os.path.dirname(os.path.abspath(__file__)),(model_name+"_trt")) 

if not os.path.exists(pb_model):
    os.mkdir(pb_model)

if not os.path.exists(trt_model):
    os.mkdir(trt_model)

tf.saved_model.save(model, pb_model)

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    rewriter_config_template=None,
    max_workspace_size_bytes=1 << 32,
    precision_mode = trt.TrtPrecisionMode.FP32,
    minimum_segment_size=50
)

#converter = trt.TrtGraphConverter(input_saved_model_dir='model.keras')
converter = trt.TrtGraphConverterV2(input_saved_model_dir=pb_model)
trt_func = converter.convert()
converter.save(trt_model)


tags = [tag_constants.SERVING]
path = saved_model_path
signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
saved_model_loaded = tf.saved_model.load(export_dir=path, tags = tags, sess = None) # path to keras .pb or TensorRT .pb
#for layer in saved_model_loaded.keras_api.layers:

graph_func = saved_model_loaded.signatures['serving_default']
frozen_func = tf.convert_variables_to_constants_V2(
    graph_func)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#convert to tensors
input_tensors = tf.cast(x_test, dtype=tf.float32)

output = frozen_func(input_tensors[:1])[0].numpy()
print(output)



#converter.summary()

'''
MAX_BATCH_SIZE = 128
def input_fn():
    batch_size = MAX_BATCH_SIZE
    x = x_test[0:batch_size, :]
    yield[x]
converter.build(input_fn = input_fn)

OUTPUT_SAVED_MODEL_DIR = './saved_models/tftrt_model'
converter.save(OUTPUT_SAVED_MODEL_DIR)

infer_batch_size = MAX_BATCH_SIZE // 2
for i in range(10):
    print(f"Step: {i}")
    start_idx = i * infer_batch_size
    end_idx = (i + 1) * infer_batch_size
    x = x_test[start_idx:end_idx, :]
    trt_func(x)
'''