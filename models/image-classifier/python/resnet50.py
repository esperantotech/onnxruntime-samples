#!/usr/bin/env python3

import numpy as np
import onnxruntime as ort
import onnx
from onnx import numpy_helper
import json
import time
import os


def load_pb_test_data_set(modelpath, test_data_num=1):
    """Load resnet pb's test dataset"""
    
    test_data_dir = os.path.join(modelpath, "test_data_set")

    #Load inputs
    inputs =  []
    for i in range(test_data_num):
        input_file = os.path.join(test_data_dir + '_{}'.format(i), 'input_0.pb')
        print(f'input_file -> {input_file}')
        tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            tensor.ParseFromString(f.read())
            inputs.append(numpy_helper.to_array(tensor))

    print('Loaded {} inputs sucessfully.'.format(test_data_num))

    # Load reference outputs
    ref_outputs = []
    for i in range(test_data_num):
        output_file = os.path.join(test_data_dir + '_{}'.format(i), 'output_0.pb')
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())    
            ref_outputs.append(numpy_helper.to_array(tensor))
        
    print('Loaded {} reference outputs successfully.'.format(test_data_num))

    print(f'inputs shape {inputs[0].shape}')
    return (inputs, ref_outputs)


def checkresults():
    """Compare result between cpu and provider"""

    print("end checkresults")


if __name__ == "__main__":
    """Launch RESNET50 onnx model over cpu and etglow and compare results."""

    modelpath = "../onnx/resnet50-v2-7/"
    modelname = "resnet50-v2-7.onnx"
    test_data_num = 1

    model = os.path.join(modelpath, modelname)
    
    inputs, ref_outputs = load_pb_test_data_set(modelpath)

    log_severity_verbose = 0
    log_severity_warning = 2
    ort.set_default_logger_severity(log_severity_verbose)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = log_severity_warning

    poptions = {}
    poptions['etglow_onnx_shape_params'] = "N=1"

    session = ort.InferenceSession(model)    
    sessionEtsoc = ort.InferenceSession(model, providers=['EtGlowExecutionProvider'], provider_options=[poptions])
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_nameEtsoc = sessionEtsoc.get_inputs()[0].name
    output_nameEtsoc = sessionEtsoc.get_outputs()[0].name

    print(f'input_name -> {input_name}')
    print(f'output_name -> {output_name}')
    print(f'input_nameEtsoc -> {input_nameEtsoc}')
    print(f'output_nameEtsoc -> {output_nameEtsoc}')

    outputs = [session.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]

    outputsEtsoc = [sessionEtsoc.run([], {input_name: inputs[i]})[0] for i in range(test_data_num)]
 

    print('Predicted {} results.'.format(len(outputs)))

    # Compare the results with reference outputs up to 4 decimal places
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, o, 4)
    
    print('ONNX Runtime outputs are similar to reference outputs!')


    print('Predicted {} Etsoc results.'.format(len(outputsEtsoc)))

    print(f'reference --> {ref_outputs}')
    print(f'resultsEtsoc --> {outputsEtsoc}')
    
    # Compare the results with reference outputs up to 4 decimal places
    for ref_o, o in zip(ref_outputs, outputsEtsoc):
        np.testing.assert_almost_equal(ref_o, o, 4)
    
    print('ONNX ETsocProvicer Runtime outputs are similar to reference outputs!')

    checkresults()