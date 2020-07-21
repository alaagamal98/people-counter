"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file or enter cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def extract_box(image, output, confidance_level=0.55):

    height  = image.shape[0]
    width = image.shape[1]
    bounding_box = output[0,0,:,3:7] * np.array([width, height , width, height ])
    bounding_box = bounding_box.astype(np.int32)
    confidance = output[0,0,:,2]
    counter=0
    p_1 = None
    p_2 = None
    for i in range(len(bounding_box)):
        if  confidance[i]<confidance_level:
            continue
        p_1 = (bounding_box[i][0], bounding_box[i][1])
        p_2 = (bounding_box[i][2], bounding_box[i][3])
        cv2.rectangle(image, p_1, p_2, (0,255,0))
        counter+=1
    return image, counter, (p_1,p_2)

def connect_mqtt():

    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    infer_network = Network()
    one_image = False

    infer_network.load_model(args.model, args.device, args.cpu_extension)
    network_shape = infer_network.get_input_shape()
    if args.input == 'CAM': 
        validator = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        one_image = True
        validator = args.input
    else:
        validator = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
        
    cap = cv2.VideoCapture(validator)
 
    if validator:
        cap.open(args.input)
    
    if (cap.isOpened()== False): 
        exit(1)
        
    total_counter=0
    pres_counter = 0
    prev_counter=0
    beginning_time=0 
    num_bounding_box=0
    timing=0
    prev_bounding_box = 0
    req_id = 0

    while cap.isOpened():
        
        flag, frame = cap.read()
        probability_threshold = args.prob_threshold
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        processed_image = cv2.resize(frame, (network_shape[3], network_shape[2]))
        processed_image = processed_image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        infer_network.exec_net(processed_image)

        if infer_network.wait(req_id) == 0:

            network_output = infer_network.get_output()
            
            frame, pres_counter, bounding_box = extract_box(frame.copy(), network_output, probability_threshold)
            box_width = frame.shape[1]
            tl, br = bounding_box 
        
            if pres_counter>prev_counter:
                beginning_time = time.time()
                total_counter+=pres_counter-prev_counter
                num_bounding_box=0
                client.publish("person", json.dumps({"total":total_counter}))

            elif pres_counter<prev_counter:
                if num_bounding_box<=20:
                    pres_counter=prev_counter
                    num_bounding_box+=1
                elif prev_bounding_box<box_width-200:
                    pres_counter=prev_counter
                    num_bounding_box=0
                else:
                    timing = int(time.time()-beginning_time)
                    client.publish("person/duration", json.dumps({"duration":timing}))

            if not (tl==None and br==None):
                prev_bounding_box=int((tl[0]+br[0])/2)
            prev_counter=pres_counter
                    
            client.publish("person", json.dumps({"count":pres_counter}))
                    
            
        frame = frame.copy(order='C')

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()


    if one_image:
        cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    client.disconnect()

def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
