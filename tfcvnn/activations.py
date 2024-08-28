import tensorflow as tf


class ComplexReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ComplexReLU, self).__init__(**kwargs)

    def call(self, inputs):
        # Apply ReLU to both real and imaginary parts
        real_part = tf.nn.relu(tf.math.real(inputs))
        imag_part = tf.nn.relu(tf.math.imag(inputs))
        return tf.complex(real_part, imag_part)
    

class ModReLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ModReLU, self).__init__(**kwargs)
        self.b = None

    def build(self, input_shape):
        # Initialize the bias parameter b
        self.b = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='modrelu_bias'
        )

    def call(self, inputs):
        # Compute the magnitude of the complex inputs
        magnitude = tf.abs(inputs)
        
        # Apply ModReLU: max(|z| + b, 0) * (z / |z|)
        scale = tf.nn.relu(magnitude + self.b) / (magnitude + 1e-9)
        scale = tf.where(tf.equal(magnitude, 0), tf.zeros_like(scale), scale)

        return inputs * tf.cast(scale, dtype=inputs.dtype)
    

class ModLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.01, **kwargs):
        """
        ModLeakyReLU activation function for complex inputs.
        Applies Leaky ReLU activation separately to the real and imaginary parts.
        
        Args:
            alpha: Slope of the negative part (leakage), typically a small number like 0.01.
        """
        super(ModLeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        # Apply Leaky ReLU to both real and imaginary parts
        real_part = tf.nn.leaky_relu(tf.math.real(inputs), alpha=self.alpha)
        imag_part = tf.nn.leaky_relu(tf.math.imag(inputs), alpha=self.alpha)
        return tf.complex(real_part, imag_part)
    

class ComplexModLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.01, **kwargs):
        """
        ModLeakyReLU activation function for complex inputs.
        Applies the Leaky ReLU activation on the modulus and preserves the phase.
        
        Args:
            alpha: Slope of the negative part (leakage), typically a small number like 0.01.
        """
        super(ComplexModLeakyReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        # Compute modulus (magnitude) of complex input
        modulus = tf.abs(inputs)

        # Apply Leaky ReLU to the modulus
        mod_leaky_relu = tf.nn.leaky_relu(modulus, alpha=self.alpha)

        # Compute the angle (phase) of the complex input
        angle = tf.math.angle(inputs)

        # Compute the new complex number using modulus after Leaky ReLU and original angle
        real_part = mod_leaky_relu * tf.math.cos(angle)
        imag_part = mod_leaky_relu * tf.math.sin(angle)
        
        return tf.complex(real_part, imag_part)
