"""
Created on Mon Jul 13 13:01:27 2020

@author: ABansal4
"""

import numpy as np
from numpy import clip
from .ie_module import Module
from .utils import resize_input

class SemanticSegmentation(Module):
    class Result:
        NUM_CLASSES = 20
        CLASSES_COLOR_MAP = [
            (150, 150, 150),
            (58, 55, 169),
            (211, 51, 17),
            (157, 80, 44),
            (23, 95, 189),
            (210, 133, 34),
            (76, 226, 202),
            (101, 138, 127),
            (223, 91, 182),
            (80, 128, 113),
            (235, 155, 55),
            (44, 151, 243),
            (159, 80, 170),
            (239, 208, 44),
            (128, 50, 51),
            (82, 141, 193),
            (9, 107, 10),
            (223, 90, 142),
            (50, 248, 83),
            (178, 101, 130),
            (71, 30, 204)
        ]

        def __init__(self, output):
            assert len(output.shape) == 4, "Expected 4D output"
            assert output.shape[0] == 1
            assert output.shape[1] == 1
            _, _, out_h, out_w = output.shape

            data = output[0]
            classes_map = np.zeros(shape=(out_h, out_w, 3), dtype=np.int)
            for i in range(out_h):
                for j in range(out_w):
                    if len(data[:, i, j]) == 1:
                        pixel_class = int(data[:, i, j])
                    else:
                        pixel_class = np.argmax(data[:, i, j])
                    classes_map[i, j, :] = self.CLASSES_COLOR_MAP[min(pixel_class, self.NUM_CLASSES)]

            self.classes_map = classes_map


    def __init__(self, model):
        super(SemanticSegmentation, self).__init__(model)

        assert len(model.inputs) == 1, "Expected 1 input blob"
        assert len(model.outputs) == 1, "Expected 1 output blob"
        self.input_blob = next(iter(model.inputs))
        self.output_blob = next(iter(model.outputs))
        self.input_shape = model.inputs[self.input_blob].shape
        self.output_shape = model.outputs[self.output_blob].shape

        assert len(self.output_shape) == 4, "Expected 4D model output" 

    def preprocess(self, frame):
        assert len(frame.shape) == 4, "Frame shape should be [1, c, h, w]"
        assert frame.shape[0] == 1
        assert frame.shape[1] == 3
        input = resize_input(frame, self.input_shape)
        return input

    def start_async(self, frame):
        input = self.preprocess(frame)
        self.enqueue(input)

    def enqueue(self, input):
        return super(SemanticSegmentation, self).enqueue({self.input_blob: input})

    def get_class_map(self):
        outputs = self.get_outputs()[0][self.output_blob]
        results = SemanticSegmentation.Result(outputs)
        return results
