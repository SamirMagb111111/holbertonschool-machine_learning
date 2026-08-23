    def style_cost(self, style_outputs):
        """Calculate the total style cost."""
        length = len(self.style_layers)

        if (not isinstance(style_outputs, list) or
                len(style_outputs) != length):
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(
                    length
                )
            )

        weight = 1 / length
        cost = 0

        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):
            cost += weight * self.layer_style_cost(
                style_output,
                gram_target
            )

        return cost
