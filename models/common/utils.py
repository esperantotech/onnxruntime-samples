#!/usr/bin/env python3

import os
import cv2
import json
import time
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional
from pathlib import Path


def load_pb_data(protobufpath : Path) -> np.ndarray:
    """Load protobuf test dataset"""
    pb_data = []
    tensor = onnx.TensorProto()
    with open(protobufpath, 'rb') as f:
        tensor.ParseFromString(f.read())
        pb_data = numpy_helper.to_array(tensor)

    return pb_data


def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def postprocess(result):
    return softmax(np.array(result)).tolist()


def get_in_out_names(session):
    """Get input and output data name of model"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return (input_name, output_name)


def check_and_compare(ref_outputs : np.ndarray, output_etsoc : np.ndarray):
    """Compare the results with reference outputs up to 4 decimal places"""
    for ref_o, o in zip(ref_outputs, output_etsoc):
        np.testing.assert_almost_equal(ref_o, o, 4)    


def test_with_protobuf(protobufpath : Path, session : ort.InferenceSession):
    """Test pb inputs given by the owner of model"""
    num_datasets = 1
    input_name = session.get_inputs()[0].name

    for i in range(num_datasets):
        input_tensor = load_pb_data(protobufpath / f'test_data_set_{i}/input_0.pb')
        output_tensor = session.run([], {input_name: input_tensor})[0]
        golden_tensor = load_pb_data(protobufpath / f'test_data_set_{i}/output_0.pb')
        check_and_compare(golden_tensor, output_tensor)


def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return data


def preprocess(input_data):
    # convert the input data to float32 datatype
    img_data = input_data.astype('float32')

    # normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        
    # Add batch dimension
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data


def load_image_files(images_path : Path) -> dict:
    """Load .png images from the specified directory"""

    if not images_path.exists():
        raise FileNotFoundError(f"The path {images_path} does not exist.")

    images = {}
    for f in images_path.iterdir():
        if f.is_file() and f.suffix == '.png':
            images[f] = {}
            img_rgb = cv2.imread(str(f))
            img_brg = np.array(img_rgb).transpose(2, 0, 1)
            images[f]['data'] = preprocess(img_brg)
            images[f]['result'] = None
            images[f]['resultcpu'] = None

    return(images)


def print_img_classification_results(labels, result, inference_time : float):
    """Print image classification results including inference execution time and confidence percentage"""

    top_idx = str(np.argmax(result))
    sorted_idx = np.flip(np.squeeze(np.argsort(result)))    
   
    print(f'Image classified as {labels[top_idx][1]}! (took {str(inference_time)}ms)')
    print('Top 5 labels:')
    for i in range(5):
        percentage = f"{result[sorted_idx[i]] * 100:.2f}%"
        print(f'    #{i+1} ({percentage}): {labels[str(sorted_idx[i])][1]}')


def test_with_tensor(tensorspath : Path, session : ort.InferenceSession):
    # TODO: Support async runs and io bindings
    output_tensors_names = [tensor.name for tensor in session.get_outputs()]

    input_tensors = {}
    for input in session.get_inputs():
        input_tensors[input.name] = np.fromfile(f'{tensorspath}/{input.name}.bin', dtype=np.int64)
        input_tensors[input.name] = input_tensors[input.name].reshape((1, 128))
        # print(f'Loaded input tensor {input.name} - shape: {input_tensors[input.name].shape}')
    
    output_tensors = session.run(output_tensors_names, input_tensors)
    # print(f'Model output tensor(s) names: {output_tensors_names}')

    return input_tensors, output_tensors


def test_with_images(imagespath : Path, session : ort.InferenceSession):
    """Test current session model against real inputs."""
    in_name, out_name = get_in_out_names(session)

    #load real images
    images = load_image_files(imagespath / 'images')
    labels = load_labels(os.path.join(imagespath / 'index_to_name.json'))

    for k,v in images.items():
        start = time.time()
        reference_output = session.run([out_name], {in_name: images[k]['data']})
        end = time.time()
        result  = postprocess(reference_output)
        inference_time = np.round((end - start) * 1000, 2)
        
        print_img_classification_results(labels, result, inference_time)
        print('==========================================')

def set_verbose_output(options, enabled):
    if not enabled:
        return
    log_severity_verbose = 0
    log_severity_warning = 2
    ort.set_default_logger_severity(log_severity_verbose)
    options.log_severity_level = log_severity_warning


def get_arg_parser(argv: Optional[Sequence[str]] = None) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("-a", "--artifacts", default="../../../DownloadArtifactory")
    parser.add_argument("-v", "--verbose", default=False)
    return parser