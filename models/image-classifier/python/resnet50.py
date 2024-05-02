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
    """Load resnet pb's test dataset"""

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

    print('Loaded {} inputs sucessfully.'.format(test_data_num))

    # Load reference outputs
    ref_outputs = []
    for i in range(test_data_num):
        output_file = os.path.join(test_data_dir + f'_{i}', 'output_0.pb')
        tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            tensor.ParseFromString(f.read())    
            ref_outputs.append(numpy_helper.to_array(tensor))
        
    print('Loaded {} reference outputs successfully.'.format(test_data_num))

    print(f'inputs shape {inputs[0].shape}')
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

    """  Currently ETGlowProvider is not working fine
    output = [sessionEtsoc.run([], {in_name: inputs[i]})[0] for i in range(test_data_num)]
    check_and_compare(ref_outputs, output)
    print('EtGlowExecutionProvider Runtime outputs are similar to reference outputs!')
    """

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return data

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def load_data_test(test_path):
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
            print("Image size: ", img.size)
            img_data = np.array(img).transpose(2, 0, 1)

            testset[f]['data'] = preprocess(img_data)
            testset[f]['result'] = None
            testset[f]['resultcpu'] = None

    return(testset)

def show_prediction(artifactsbase, idx, res, inference_time):
    """Show ResNet result"""

    labels = load_labels(f'{artifactsbase}/input_data/images/imagenet/index_to_name.json')
   
    print('========================================')
    print('Final top prediction is: ' + labels[f'{idx}'][1])
    print('========================================')

    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')

    sort_idx = np.flip(np.squeeze(np.argsort(res)))       
    print('============ Top 5 labels are: ============================')
    for i in sort_idx[:5]:
        print(labels[f'{i}'][1])
    print('===========================================================')


def test_images(modelpath, session, sessionEtsoc):
    """Test current session model against real inputs."""

    artifactsbase = '/'.join(modelpath.split('/')[:-2])
    test_path = [f'{artifactsbase}/input_data/images/imagenet/images']

    in_name, out_name = get_in_out_names(session)

    #load real images
    testset = load_data_test(test_path)    

    for k,v in testset.items():
        start = time.time()
        result = session.run([out_name], {in_name: testset[k]['data']})
        end = time.time()
        res  = postprocess(result)
        inference_time = np.round((end - start) * 1000, 2)        
        idx = np.argmax(res)
        show_prediction(artifactsbase, idx, res, inference_time)

        """  Currently ETGlowProvider is not working fine
        start = time.time()
        result = sessionEtsoc.run([out_name], {in_name: testset[k]['data']})
        end = time.time()
        res  = postprocess(result)
        inference_time = np.round((end - start) * 1000, 2)        
        idx = np.argmax(res)
        show_prediction(idx, res, inference_time)
        """

def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory")

    return parser.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None):
    """Launch RESNET50 onnx model over cpu and etglow and compare results."""

    args = parse_args(argv)    
    #modelpath = args.artifacts + "/models/resnet50_denso_onnx"
    #modelpath = args.artifacts + "/models/resnet50_onnx"    
    modelpath = args.artifacts + "/models/resnet50-v2-7"
    modelname = "model.onnx"
    model = os.path.join(modelpath, modelname)
    
    log_severity_verbose = 0
    log_severity_warning = 2
    ort.set_default_logger_severity(log_severity_verbose)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = log_severity_warning

    poptions = {}
    poptions['etglow_onnx_shape_params'] = "N=1"

    session = ort.InferenceSession(model)    
    sessionEtsoc = ort.InferenceSession(model, providers=['EtGlowExecutionProvider'], provider_options=[poptions])

    test_pb_inputs(modelpath, session, sessionEtsoc)
    test_images(modelpath, session, sessionEtsoc)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
