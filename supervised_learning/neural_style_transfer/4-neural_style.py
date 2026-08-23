    def layer_style_cost(self, style_output, gram_target):
        """Calculate the style cost for a single layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(
                "style_output must be a tensor of rank 4"
            )

        c = style_output.shape[-1]

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                len(gram_target.shape) != 3 or
                gram_target.shape[0] != 1 or
                gram_target.shape[1] != c or
                gram_target.shape[2] != c):
            raise TypeError(
                "gram_target must be a tensor of shape "
                "[1, {}, {}]".format(c, c)
            )

        gram_style = self.gram_matrix(style_output)

        cost = tf.reduce_sum(
            tf.square(gram_style - gram_target)
        )

        cost = cost / tf.cast(c ** 2, tf.float32)

        return cost
