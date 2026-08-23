#!/usr/bin/env python3
"""YOLOv3 object detection module."""

import cv2
import glob
import os
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

    def non_max_suppression(
            self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered bounding boxes."""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for class_num in np.unique(box_classes):
            mask = box_classes == class_num

            class_boxes = filtered_boxes[mask]
            class_scores = box_scores[mask]

            while len(class_boxes) > 0:
                max_index = np.argmax(class_scores)

                best_box = class_boxes[max_index]
                best_score = class_scores[max_index]

                box_predictions.append(best_box)
                predicted_box_classes.append(class_num)
                predicted_box_scores.append(best_score)

                class_boxes = np.delete(
                    class_boxes, max_index, axis=0
                )
                class_scores = np.delete(
                    class_scores, max_index, axis=0
                )

                if len(class_boxes) == 0:
                    break

                x1 = np.maximum(best_box[0], class_boxes[:, 0])
                y1 = np.maximum(best_box[1], class_boxes[:, 1])
                x2 = np.minimum(best_box[2], class_boxes[:, 2])
                y2 = np.minimum(best_box[3], class_boxes[:, 3])

                intersection_width = np.maximum(0, x2 - x1)
                intersection_height = np.maximum(0, y2 - y1)

                intersection = (
                    intersection_width * intersection_height
                )

                best_area = (
                    (best_box[2] - best_box[0])
                    * (best_box[3] - best_box[1])
                )

                box_areas = (
                    (class_boxes[:, 2] - class_boxes[:, 0])
                    * (class_boxes[:, 3] - class_boxes[:, 1])
                )

                union = best_area + box_areas - intersection
                iou = intersection / union

                keep = iou < self.nms_t

                class_boxes = class_boxes[keep]
                class_scores = class_scores[keep]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return (
            box_predictions,
            predicted_box_classes,
            predicted_box_scores
        )

    @staticmethod
    def load_images(folder_path):
        """Load all images from a folder."""
        images = []
        image_paths = []

        for image_path in glob.glob(folder_path + "/*"):
            image = cv2.imread(image_path)

            if image is not None:
                images.append(image)
                image_paths.append(image_path)

        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images for the YOLO model."""
        pimages = []
        image_shapes = []

        input_height = self.model.input.shape[1]
        input_width = self.model.input.shape[2]

        for image in images:
            image_shapes.append(
                [image.shape[0], image.shape[1]]
            )

            resized_image = cv2.resize(
                image,
                (input_width, input_height),
                interpolation=cv2.INTER_CUBIC
            )

            resized_image = resized_image / 255.0

            pimages.append(resized_image)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(
            self, image, boxes, box_classes, box_scores, file_name):
        """Display an image with detected bounding boxes."""
        for box, class_index, score in zip(
                boxes, box_classes, box_scores):

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            class_name = self.class_names[class_index]
            text = "{} {:.2f}".format(class_name, score)

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)

        if key == ord("s"):
            if not os.path.exists("detections"):
                os.makedirs("detections")

            save_path = os.path.join(
                "detections",
                os.path.basename(file_name)
            )

            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()
