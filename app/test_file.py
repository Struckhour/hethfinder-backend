

import tensorflow as tf

model = tf.saved_model.load("app/ml/ml_songs_v1.model")

infer = model.signatures["serving_default"]

# example input (shape must match your model)
import numpy as np

x = np.random.rand(1, 623, 69, 1).astype(np.float32)

out = infer(conv2d_input=tf.constant(x))

print(out)