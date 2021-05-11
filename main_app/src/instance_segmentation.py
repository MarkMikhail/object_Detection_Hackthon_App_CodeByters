"""
Created on Mon Jul 13 13:01:27 2020

@author: ABansal4
"""

import numpy as np
import cv2

from .ie_module import Module
from .utils import resize_input
from .color import COLOR_PALETTE


def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp

def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask


class InstanceSegmentation(Module):
    class Result:

        def __init__(self, box, cls, cls_id, score, mask):
            box = box.astype(int)

            self.label = cls
            self.position = box[:2]
            self.size = box[2:] - box[:2]
            self.confidence = score
            self.mask = mask
            self.color = COLOR_PALETTE[cls_id]


    def __init__(self, model, labels_file, confidence_threshold=0.5):
        super(InstanceSegmentation, self).__init__(model)

        assert len(model.inputs) == 2, "Expected 2 input blob"
        assert len(model.outputs) == 4, "Expected 4 output blob"
        
        input_iter = iter(model.inputs)
        self.im_data_blob = next(input_iter)
        self.im_info_blob = next(input_iter)
        
        output_iter = iter(model.outputs)
        self.bboxes_blob = next(output_iter)
        self.classes_blob = next(output_iter)
        self.raw_masks_blob = next(output_iter)
        self.scores_blob = next(output_iter)

        self.input_shape = model.inputs[self.im_data_blob].shape
        assert len(model.outputs[self.classes_blob].shape) == 1, f"Expected 1D arrary for class output {model.outputs[self.classes_blob].shape}"
        assert len(model.outputs[self.scores_blob].shape) == 1, "Expected 1D arrary for Scores output"
        assert len(model.outputs[self.bboxes_blob].shape) == 2, "Expected 2D arrary for Bounding Box output"
        assert len(model.outputs[self.raw_masks_blob].shape) == 4, "Expected 4D arrary for raw_masks output"
        
        assert 0.0 <= confidence_threshold and confidence_threshold <= 1.0, \
            "Confidence threshold is expected to be in range [0, 1]"
        self.confidence_threshold = confidence_threshold

        with open(labels_file, 'rt') as labels_file:
            self.class_labels = labels_file.read().splitlines()

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        self.frame_w = frame.shape[3]
        self.frame_h = frame.shape[2]
        self.scale_x = input.shape[3] / self.frame_w
        self.scale_y = input.shape[2] / self.frame_h

        return input

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        # Not maintaining the aspect ratio for now
        im_info = np.asarray([*self.input_shape[-2:], 1.])
        return super(InstanceSegmentation, self).enqueue({self.im_data_blob: input, self.im_info_blob: im_info})

    def object_masks(self):
        outputs = self.get_outputs()[0]
        # outputs shape is [N_requests]
        results = []
        # Parse detection results of the current request
        classes = outputs[self.classes_blob].astype(np.uint32)
        scores = outputs[self.scores_blob]
        boxes = outputs[self.bboxes_blob]
        boxes[:, 0::2] /= self.scale_x
        boxes[:, 1::2] /= self.scale_y

        for box, cls, score, raw_mask in zip(boxes, classes, scores, outputs[self.raw_masks_blob]):
            # Filter out detections with low confidence.
            if score < self.confidence_threshold:
                continue
            raw_cls_mask = raw_mask[cls, ...]
            mask = segm_postprocess(box, raw_cls_mask, self.frame_h, self.frame_w)        
            result = InstanceSegmentation.Result(box, self.class_labels[cls], cls, score, mask)
            results.append(result)

        return results
