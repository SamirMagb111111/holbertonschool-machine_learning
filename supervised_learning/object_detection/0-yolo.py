#!/usr/bin/env python3
"""Yolo object detection module."""

import tensorflow.keras as K


class Yolo:
    """Class that uses YOLOv3 to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialize a YOLOv3 object detector.

        Args:
            model_path: Path to the Darknet Keras model.
            classes_path: Path to the file containing class names.
            class_t: Box score threshold for initial filtering.
            nms_t: IOU threshold for non-max suppression.
            anchors: Array containing anchor box dimensions.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r", encoding="utf-8") as file:
            self.class_names = [line.strip() for line in file]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
