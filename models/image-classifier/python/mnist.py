#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import json
import time
import os
import sys
import cv2
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional


def load_pb_data(modelpath, test_data_num=1):
    """Load minst pb's test dataset"""
    
    protobufbasepath = "input_data/protobuf"    
    modelname = modelpath.split('/')[-1]
    artifactsbase = '/'.join(modelpath.split('/')[:-2])

    protobufmodelpath = os.path.join(artifactsbase, protobufbasepath, modelname)

    test_data_dir = os.path.join(protobufmodelpath, "test_data_set")
        
    #Load inputs
    inputs =  []
    for i in range(test_data_num):
        input_file = os.path.join(test_data_dir + f'_{i}', 'input_0.pb')
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))

    print('Loaded {test_data_num} inputs sucessfully.')

    # Load reference outputs
    ref_outputs = []
    for i in range(test_data_num):
        output_file = os.path.join(test_data_dir + f'_{i}', 'output_0.pb')
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())    
            ref_outputs.append(numpy_helper.to_array(tensor))
        
    print('Loaded {test_data_num} reference outputs successfully.')

    return (inputs, ref_outputs)

def get_in_out_names(session):
    """Get input and output data name of model"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return (input_name, output_name)

def check_and_compare(ref_outputs, outputEtsoc) :
    """Compare the results with reference outputs up to 4 decimal places"""
   
    for ref_o, o in zip(ref_outputs, outputEtsoc):
        np.testing.assert_almost_equal(ref_o, o, 4)        

def test_pb_inputs(modelpath, session, sessionEtsoc):
    """Test pb inputs given by the owner of model"""
    test_data_num = 1

    in_name, out_name = get_in_out_names(session)
    inputs, ref_outputs = load_pb_data(modelpath, test_data_num)

    output = [session.run([], {in_name: inputs[i]})[0] for i in range(test_data_num)]
    check_and_compare(ref_outputs, output)
    print('CPU Runtime outputs are similar to reference outputs!')
    
    output = [sessionEtsoc.run([], {in_name: inputs[i]})[0] for i in range(test_data_num)]
    check_and_compare(ref_outputs, output)
    print('EtGlowExecutionProvider Runtime outputs are similar to reference outputs!')

def load_mnist_test(test_path):
    """Load images to be tested"""

    extension = '.png'  

    im_files=[]
    im_files_path=[]

    for i in test_path:
        im_files = os.listdir(i)
        for j in im_files:
            im_files_path.append(os.path.join(i, j))

    testset = {}

    for f in im_files_path:
        if f.endswith(extension):
            testset[f] = {}
            img = cv2.imread(f)
            #When translating a color image to grayscale (mode “L”), the library uses the ITU-R 601-2 luma transform
            #L = R * 299/1000 + G * 587/1000 + B * 114/1000
            img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
            img = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_AREA).astype(np.float32)
            img.resize((1,1,28,28))    
            testset[f]['data'] = img
            testset[f]['result'] = None
            testset[f]['resultcpu'] = None

    return(testset)

def checkresults(testset):
    """Compare result between cpu and provider"""
    
    for k, v in testset.items():
        if testset[k]['result'] == testset[k]['resultcpu'] :
            print(f'Image {k} is OK \n\tidentified as number {testset[k]["resultcpu"]}')
        else:
            print("****************")
            print(f'Image {k} FAILS:')
            print(f'\tExpected result is {testset[k]["result"]} \n\tobtained in provider {testset[k]["resultcpu"]}')
            print("****************")

def test_images(modelpath, session, sessionEtsoc):
    """Test current session model against real inputs."""

    artifactsbase = '/'.join(modelpath.split('/')[:-2])
    test_path = [f'{artifactsbase}/input_data/images/images_1-1-28-28',
                 f'{artifactsbase}/input_data/images/images_3-400-640']
    
    in_name, out_name = get_in_out_names(session)

    #load real images
    testset = load_mnist_test(test_path)    
    for k,v in testset.items():
        result = session.run([out_name], {in_name: testset[k]['data']})
        testset[k]['result'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

        result = sessionEtsoc.run([out_name], {in_name: testset[k]['data']})
        testset[k]['resultcpu'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

    checkresults(testset)

def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory")

    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None):    
    """Launch MNIST onnx model over cpu and etglow and compare results."""

    args = parse_args(argv)        
    modelpath = args.artifacts + "/models/mnist"
    modelname = "model.onnx"
    model = os.path.join(modelpath, modelname)
    print(modelpath)

    session = ort.InferenceSession(model)
    sessionEtsoc = ort.InferenceSession(model, providers=['EtGlowExecutionProvider'])

    test_pb_inputs(modelpath, session, sessionEtsoc)
    test_images(modelpath, session, sessionEtsoc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
