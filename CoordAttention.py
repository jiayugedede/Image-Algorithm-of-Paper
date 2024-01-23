import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, name="HardSigmoid", **kwargs):
        super(HardSigmoid, self).__init__(name=name)
        super(HardSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name=f"ReLU6_{h}x{w}")
        super().build(input_shape)

    def call(self, input):
        return input * self.relu6(input + 3.0) / 6.0

    def get_config(self):
        base_config = super(HardSigmoid, self).get_config()
        return dict(list(base_config.items()))


@tf.keras.utils.register_keras_serializable()
class HardSwish(tf.keras.layers.Layer):
    def __init__(self, name="HardSwish", **kwargs):
        super(HardSwish, self).__init__(name=name)
        super(HardSwish, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.sigmoid = HardSigmoid(name=f"h_sigmoid_{h}x{w}")
        super().build(input_shape)

    def call(self, input):
        output = input * self.sigmoid(input)
        return output

    def get_config(self):
        base_config = super(HardSwish, self).get_config()
        return dict(list(base_config.items()))


@tf.keras.utils.register_keras_serializable()
class CoordAttention(tf.keras.layers.Layer):
    def __init__(self, output_channels, name, groups=32, **kwargs):
        super(CoordAttention, self).__init__(name=name)
        self.output_channels = output_channels
        self.groups = groups
        super(CoordAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.mip = max(8, c // self.groups)

        self.pool_h = tf.keras.layers.AveragePooling2D(pool_size=(1, w),
                                                       name=f"average_pooling_h_1_{h}_{w}")
        self.pool_w = tf.keras.layers.AveragePooling2D(pool_size=(h, 1),
                                                       name=f"average_pooling_1_h_{h}_{w}")
        self.permute_w = tf.keras.layers.Permute(dims=[0, 2, 1, 3], name=f"permute_w_{h}x{w}")

        self.permute_x_w = tf.keras.layers.Permute(dims=[0, 2, 1, 3], name=f"permute_x_w_{h}x{w}")

        self.concat = tf.keras.layers.Concatenate(axis=1, name=f"concate_h_w{h}x{w}")

        self.conv1 = tf.keras.layers.Conv2D(self.mip, kernel_size=1,
                                            use_bias=True, strides=(1, 1),
                                            padding='valid',
                                            name=f"conv_1_{h}x{w}")
        self.conv_h = tf.keras.layers.Conv2D(self.output_channels, kernel_size=1,
                                             use_bias=True, strides=(1, 1),
                                             padding='valid',
                                             name=f"conv_h_{h}x{w}")
        self.conv_w = tf.keras.layers.Conv2D(self.output_channels, kernel_size=1,
                                             use_bias=True, strides=(1, 1),
                                             padding='valid',
                                             name=f"conv_w_{h}x{w}")
        self.bn1 = tf.keras.layers.BatchNormalization(name=f"bn_1_{h}x{w}")
        self.act = HardSwish(name=f"act_{h}x{w}")

        self.sigmoid_a_h = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=f"sigmoid_a_h_{h}x{w}")
        self.sigmoid_a_w = tf.keras.layers.Activation(tf.keras.activations.sigmoid, name=f"sigmoid_a_w_{h}x{w}")
        super().build(input_shape)

    def split(self, inputs):
        split_1, split_2 = tf.split(inputs, num_or_size_splits=2, axis=1)
        return split_1, split_2

    def call(self, inputs):
        identity = inputs
        x_h = self.pool_h(inputs)
        x_w = self.pool_w(inputs)
        x_w = self.permute_w(x_w)
        y = self.concat([x_h, x_w])
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = self.split(y)
        x_w = self.permute_x_w(x_w)
        a_h = self.conv_h(x_h)
        a_h = self.sigmoid_a_h(a_h)
        a_w = self.conv_w(x_w)
        a_w = self.sigmoid_a_w(a_w)
        output = identity * a_w * a_h
        return output

    def get_config(self):
        config = {"output_channels": self.output_channels,
                  "groups": self.groups}
        base_config = super(CoordAttention, self).get_config()
        return dict(list(base_config.items() + list(config.items())))
