import tensorflow as tf


class ComplexDense(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(ComplexDense, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialize weights for the real and imaginary parts
        self.real_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='imag_kernel'
        )
        self.real_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='real_bias'
        )
        self.imag_bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='imag_bias'
        )

    def call(self, inputs):
        real_input = tf.math.real(inputs)
        imag_input = tf.math.imag(inputs)

        real_part = tf.matmul(real_input, self.real_kernel) - tf.matmul(imag_input, self.imag_kernel)
        imag_part = tf.matmul(real_input, self.imag_kernel) + tf.matmul(imag_input, self.real_kernel)

        real_part = real_part + self.real_bias
        imag_part = imag_part + self.imag_bias

        return tf.complex(real_part, imag_part)
    

class ComplexConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Initialize weights for the real and imaginary parts of the kernel
        self.real_kernel = self.add_weight(
            shape=(*self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=(*self.kernel_size, input_shape[-1], self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='imag_kernel'
        )
        self.real_bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='real_bias'
        )
        self.imag_bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='imag_bias'
        )

    def call(self, inputs):
        # Extract real and imaginary parts from the input
        real_input = tf.math.real(inputs)
        imag_input = tf.math.imag(inputs)

        # Perform convolutions separately for real and imaginary parts
        real_conv = tf.nn.conv2d(real_input, self.real_kernel, strides=self.strides, padding=self.padding.upper())
        imag_conv = tf.nn.conv2d(imag_input, self.imag_kernel, strides=self.strides, padding=self.padding.upper())
        real_conv += tf.nn.conv2d(imag_input, self.imag_kernel, strides=self.strides, padding=self.padding.upper())
        imag_conv -= tf.nn.conv2d(real_input, self.imag_kernel, strides=self.strides, padding=self.padding.upper())

        # Add biases
        real_conv += self.real_bias
        imag_conv += self.imag_bias

        # Combine real and imaginary parts
        output = tf.complex(real_conv, imag_conv)
        return output
    

class ComplexConv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', **kwargs):
        super(ComplexConv2DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Initialize weights for the real and imaginary parts of the kernel
        self.real_kernel = self.add_weight(
            shape=(*self.kernel_size, self.filters, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='real_kernel'
        )
        self.imag_kernel = self.add_weight(
            shape=(*self.kernel_size, self.filters, input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True,
            name='imag_kernel'
        )
        self.real_bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='real_bias'
        )
        self.imag_bias = self.add_weight(
            shape=(self.filters,),
            initializer='zeros',
            trainable=True,
            name='imag_bias'
        )

    def call(self, inputs):
        # Extract real and imaginary parts from the input
        real_input = tf.math.real(inputs)
        imag_input = tf.math.imag(inputs)

        # # Compute output shape
        # output_shape = self._compute_output_shape(inputs.shape)

        # Perform transposed convolutions separately for real and imaginary parts
        real_conv = tf.nn.conv2d_transpose(
            real_input, self.real_kernel, output_shape=output_shape, strides=self.strides, padding=self.padding.upper()
        )
        imag_conv = tf.nn.conv2d_transpose(
            imag_input, self.imag_kernel, output_shape=output_shape, strides=self.strides, padding=self.padding.upper()
        )
        real_conv += tf.nn.conv2d_transpose(
            imag_input, self.imag_kernel, output_shape=output_shape, strides=self.strides, padding=self.padding.upper()
        )
        imag_conv -= tf.nn.conv2d_transpose(
            real_input, self.imag_kernel, output_shape=output_shape, strides=self.strides, padding=self.padding.upper()
        )

        # Add biases
        real_conv += self.real_bias
        imag_conv += self.imag_bias

        # Combine real and imaginary parts
        output = tf.complex(real_conv, imag_conv)
        return output

    def _compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        kernel_height, kernel_width = self.kernel_size
        strides_height, strides_width = self.strides

        if self.padding.upper() == 'SAME':
            output_height = height * strides_height
            output_width = width * strides_width
        elif self.padding.upper() == 'VALID':
            output_height = (height - 1) * strides_height + kernel_height
            output_width = (width - 1) * strides_width + kernel_width
        else:
            raise ValueError("Padding type should be either 'same' or 'valid'")

        return [None, output_height, output_width, self.filters]