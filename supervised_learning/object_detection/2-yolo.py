#!/usr/bin/env python3
"""YOLOv3 object detection module."""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Class that uses YOLOv3 to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the YOLOv3 object detector."""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as file:
            self.class_names = [line.strip() for line in file]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process the outputs of the Darknet model."""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height = output.shape[0]
            grid_width = output.shape[1]

            box = output[..., :4].copy()

            grid_y, grid_x = np.indices((grid_height, grid_width))
            grid_x = np.expand_dims(grid_x, axis=-1)
            grid_y = np.expand_dims(grid_y, axis=-1)

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            bx = (
                (1 / (1 + np.exp(-tx)) + grid_x)
                / grid_width
            )

            by = (
                (1 / (1 + np.exp(-ty)) + grid_y)
                / grid_height
            )

            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]

            bw = (
                np.exp(tw) * anchor_width
                / input_width
            )

            bh = (
                np.exp(th) * anchor_height
                / input_height
            )

            box[..., 0] = (bx - bw / 2) * image_width
            box[..., 1] = (by - bh / 2) * image_height
            box[..., 2] = (bx + bw / 2) * image_width
            box[..., 3] = (by + bh / 2) * image_height

            box_confidence = (
                1 / (1 + np.exp(-output[..., 4:5]))
            )

            box_class_prob = (
                1 / (1 + np.exp(-output[..., 5:]))
            )

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes based on their class scores."""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):

            scores = confidence * class_probs

            classes = np.argmax(scores, axis=-1)
            scores_max = np.max(scores, axis=-1)

            mask = scores_max >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(scores_max[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
