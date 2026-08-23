    def generate_features(self):
        """Extract style and content features."""
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(style_image)
        content_outputs = self.model(content_image)

        self.gram_style_features = [
            self.gram_matrix(output)
            for output in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]
