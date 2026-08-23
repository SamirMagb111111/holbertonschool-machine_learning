    @staticmethod
    def gram_matrix(input_layer):
        """Calculate the Gram matrix of a layer output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum(
            'bijc,bijd->bcd',
            input_layer,
            input_layer
        )

        height = tf.shape(input_layer)[1]
        width = tf.shape(input_layer)[2]

        gram = gram / tf.cast(
            height * width,
            tf.float32
        )

        return gram
