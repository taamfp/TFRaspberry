### TLite Raspberry Pi model converter ###

import os
import tensorflow as tf

model_direct = 'path'
model = tf.keras.models.load_model(model_direct)
tf_lite_model = 'model_tflite.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tf_lite_model, 'wb') as f:
  f.write(tflite_model)