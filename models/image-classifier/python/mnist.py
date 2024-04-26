#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import json
import time
import os
import cv2


def load_pb_test_data_set(modelpath, test_data_num=1):
    """Load minst pb's test dataset"""
    
    test_data_dir = os.path.join(modelpath, "test_data_set")
        
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
            print(f'Image {k} Ok \n\tidentified as number {testset[k]["resultcpu"]}')
        else:
            print("****************")
            print(f'Image {k} FAILS:')
            print(f'\tExpected result is {testset[k]["result"]} \n\tobtained in provider {testset[k]["resultcpu"]}')
            print("****************")

if __name__ == "__main__":
    """Launch mnist onns model over cpu and etglow and compare results."""

    modelpath = "../onnx/mnist-12"
    modelname = "mnist-12.onnx"
    test_path = ['../images/images_1-1-28-28',
                 '../images/images_3-400-640']
    test_data_num = 1

    model = os.path.join(modelpath, modelname)

    session = ort.InferenceSession(model)
    sessionEtsoc = ort.InferenceSession(model, providers=['EtGlowExecutionProvider'])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    inputs, ref_outputs = load_pb_test_data_set(modelpath)

    outputsEtsoc = [sessionEtsoc.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

    # Compare the results with reference outputs up to 4 decimal places
    for ref_o, o in zip(ref_outputs, outputsEtsoc):
        np.testing.assert_almost_equal(ref_o, o, 4)    
    print('ONNX EtGlowExecutionProvider Runtime outputs are similar to reference outputs!')

    #load real images
    testset = load_mnist_test(test_path)    
    for k,v in testset.items():
        result = session.run([output_name], {input_name: testset[k]['data']})
        testset[k]['result'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

        result = sessionEtsoc.run([output_name], {input_name: testset[k]['data']})
        testset[k]['resultcpu'] = int(np.argmax(np.array(result).squeeze(), axis = 0))

    checkresults(testset)