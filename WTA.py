import tensorflow as tf

class WTA(tf.keras.layers.Layer):
    def __init__(self, name, **kwargs):
        super(WTA, self).__init__(name=name)
        super(WTA, self).__init__(**kwargs)

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.relu = tf.keras.layers.ReLU(name="WTA", name=f"WTA_ReLU_{h}x{w}")
        self.coord_attention = CoordAttention(output_channels=c, name=f"WTA_coorda_{h}x{w}")
        self.addition = tf.keras.layers.Add(name=f"WTA_addition_{h}x{w}")
        super().build(input_shape)

    def call(self, input_tensor):
        Max_value_tensor = tf.math.reduce_max(input_tensor=input_tensor, axis=[1,2], keepdims=True)
        temp_value = input_tensor / Max_value_tensor
        processed_tensor = temp_value - 0.9999999
        winner_take_all = winner_take_all* 10000000 * Max_value_tensor
        enhanced = self.coord_attention(winner_take_all)
        output = self.addition([winner_take_all, enhanced])
        return output

    def get_config(self):
        base_config = super(WTA, self).get_config()
        return dict(list(base_config.items()))
