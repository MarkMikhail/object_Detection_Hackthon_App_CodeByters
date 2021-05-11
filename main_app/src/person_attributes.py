"""
Created on Wed Jul 8 13:01:53 2020

@author: ARawat4
"""

import numpy as np

from .utils import cut_rois, resize_input
from .ie_module import Module

class PersonAttributes(Module):
    class Result:
        def __init__(self, attributes, top_color, bottom_color):
            att_keys = ["is_male", "has_bag", "has_backpack", "has_hat", "has_longsleeves",
                         "has_longpants", "has_longhair", "has_coat_jacket"]
            attributes = attributes.flatten()
            self.attributes = dict(zip(att_keys, attributes))
            self.top_color_point = top_color.flatten()
            self.top_color_point = bottom_color.flatten()

    def __init__(self, model):
        super(PersonAttributes, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 3, "Expected 3 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_iter = iter(model.outputs)
        self.attribute_blob = next(self.output_iter)
        self.top_color_blob = next(self.output_iter)
        self.bottom_color_blob = next(self.output_iter)
        self.input_shape = model.inputs[self.input_blob].shape

    def preprocess(self, frame, rois):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(PersonAttributes, self).enqueue({self.input_blob: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def get_attributes(self):
        outputs = self.get_outputs()
        results = [PersonAttributes.Result(out[self.attribute_blob], out[self.top_color_blob], out[self.bottom_color_blob]) \
                      for out in outputs]
        return results
