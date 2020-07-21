#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):

        self.plugin = None
        self.network = None
        self.exec_network = None
        self.input_layer = None
        self.output_layer = None

    def load_model(self, model, device="CPU", cpu_extension=None):

        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        self.plugin = IECore()
        
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            
        self.network = IENetwork(model=model_xml, weights=model_bin)
        
        supported_layers = self.plugin.query_network(self.network, device)
        present_layers = self.network.layers.keys()
        
        for layer in present_layers:
            if layer not in supported_layers:
                print('Layer ' + layer + 'is not supported')
                exit(1)
      
        self.exec_network = self.plugin.load_network(self.network, device)
        
        self.input_layer = next(iter(self.network.inputs))
        self.output_layer = next(iter(self.network.outputs))
        
        return self.plugin

    def get_input_shape(self):
        
        return self.network.inputs[self.input_layer].shape

    def exec_net(self, image):

        self.inference_request = self.exec_network.start_async(0, {self.input_layer: image})

        return self.inference_request

    def wait(self, request_id):
 
        return self.exec_network.requests[request_id].wait(-1)

    def get_output(self):
 
        output = self.inference_request.outputs[self.output_layer]
        return output