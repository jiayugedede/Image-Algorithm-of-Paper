import tensorflow as tf

def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)

Conv2D(filters=16, kernel_size=5, padding="SAME", kernel_regularizer=ws_reg)