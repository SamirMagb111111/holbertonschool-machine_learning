    def content_cost(self, content_output):
        """Calculate the content cost."""
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(
                    self.content_feature.shape
                )
            )

        cost = tf.reduce_mean(
            tf.square(content_output - self.content_feature)
        )

        return cost
