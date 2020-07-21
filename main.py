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
import logging as log
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
                        help="Path to image or video file")
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
    parser.add_argument("-pt", "--probability_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


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
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    probability_threshold = args.probability_threshold
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
    width = int(cap.get(3))
    height = int(cap.get(4))
    if validator:
        cap.open(args.input)
    if not cap.isOpened():
        log.error("Error: No video source")
    
    ### Variables
    previous_duration = 0
    dur = 0
    req_id=0
    dur_report = None
    report = 0
    counter = 0
    previous_counter = 0
    total_counter = 0

    while cap.isOpened():

        flag, frame = cap.read()
        probability_threshold = args.probability_threshold
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        processed_image = cv2.resize(frame, (network_shape[3], network_shape[2]))
        processed_image = processed_image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        infer_network.exec_net(processed_image)

        if infer_network.wait(req_id) == 0:

            network_output = infer_network.get_output()

            check = 0
            probabilities = network_output[0, 0, :, 2]
            for i, p in enumerate(probabilities):
                if p > probability_threshold:
                    check += 1
                    box = network_output[0, 0, i, 3:]
                    p_1 = (int(box[0] * width), int(box[1] * height))
                    p_2 = (int(box[2] * width), int(box[3] * height))
                    frame = cv2.rectangle(frame, p_1, p_2, (0, 255, 0), 3)
        
            if check != counter:
                previous_counter = counter
                counter = check
                if dur >= 3:
                    previous_duration = dur
                    dur = 0
                else:
                    dur = previous_duration + dur
                    previous_duration = 0  
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > previous_counter:
                        total_counter += counter - previous_counter
                    elif dur == 3 and counter < previous_counter:
                        dur_report = int((previous_duration / 10.0) * 1000)

            client.publish('person',
                           payload=json.dumps({
                               'count': report, 'total': total_counter}),
                           qos=0, retain=False)
            if dur_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': dur_report}),
                               qos=0, retain=False)

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if one_image:
            cv2.imwrite('output_image.jpg', frame)
            
    cap.release()
    cv2.destroyAllWindows()
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
